"""Small feed-forward implementations; discrete actions, parameter-shared local agents.

IQL/VDN/QMIX/QTRAN: one-step TD with target networks.
IAC/IPPO/MAPPO: bootstrapped rollout returns, separate local/central critics.
COMA: action-conditioned centralized critic and counterfactual baseline.
PPO: one centralized policy choosing a vector of categorical actions.
"""
import copy
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


def network(inputs,outputs):
    return nn.Sequential(nn.Linear(inputs,32),nn.Tanh(),nn.Linear(32,outputs))


class Mixer(nn.Module):
    def __init__(self,n,state):
        super().__init__(); self.n=n
        self.w1=nn.Linear(state,n*16); self.b1=nn.Linear(state,16)
        self.w2=nn.Linear(state,16); self.b2=network(state,1)

    def forward(self,q,s):
        w1=self.w1(s).abs().reshape(-1,self.n,16)
        hidden=nn.functional.elu(torch.bmm(q[:,None,:],w1).squeeze(1)+self.b1(s))
        return (hidden*self.w2(s).abs()).sum(1)+self.b2(s).squeeze(1)


class Agent(nn.Module):
    def __init__(self,name,n,obs,lr=.001,gamma=.95):
        super().__init__(); self.name=name; self.n=n; self.obs=obs; self.gamma=gamma; self.updates=0
        self.q_based=name in ['IQL','VDN','QMIX','QTRAN']
        self.local=network(obs,5)
        self.value=network(n*obs if name in ['MAPPO','PPO'] else obs,1)
        if name=='PPO': self.central_actor=network(n*obs,n*5)
        if name=='COMA': self.counterfactual=network(n*obs+n*5+n,n*0+5)
        if name=='QMIX': self.mixer=Mixer(n,n*obs)
        if name=='QTRAN': self.joint=network(n*obs+n*5,1); self.offset=network(n*obs,1)
        self.optimizer=torch.optim.Adam(self.parameters(),lr=lr)
        self.target_local=copy.deepcopy(self.local).requires_grad_(False)
        if name=='QMIX': self.target_mixer=copy.deepcopy(self.mixer).requires_grad_(False)
        if name=='QTRAN': self.target_joint=copy.deepcopy(self.joint).requires_grad_(False)

    def logits(self,obs):
        if self.name=='PPO': return self.central_actor(obs.flatten(-2)).reshape(-1,self.n,5)
        return self.local(obs)

    def act(self,obs,explore=True,epsilon=.1):
        x=torch.as_tensor(obs,dtype=torch.float32)[None]
        with torch.no_grad():
            logits=self.logits(x)
            if self.q_based:
                action=logits.argmax(-1)
                if explore:
                    choose=torch.rand_like(action.float())<epsilon
                    action=torch.where(choose,torch.randint(5,action.shape),action)
                return action[0].numpy(),np.zeros(self.n,dtype=np.float32)
            distribution=Categorical(logits=logits)
            action=distribution.sample() if explore else logits.argmax(-1)
            return action[0].numpy(),distribution.log_prob(action)[0].numpy()

    def coma_q(self,obs,actions):
        batch=len(obs); onehot=nn.functional.one_hot(actions,5).float()
        state=obs.flatten(1)
        qs=[]
        for i in range(self.n):
            others=onehot.clone(); others[:,i]=0
            identity=torch.zeros(batch,self.n); identity[:,i]=1
            qs.append(self.counterfactual(torch.cat([state,others.flatten(1),identity],1)))
        return torch.stack(qs,1)

    def update(self,transitions):
        obs,actions,rewards,next_obs,dones,old_logp=zip(*transitions)
        o=torch.tensor(np.stack(obs),dtype=torch.float32); no=torch.tensor(np.stack(next_obs),dtype=torch.float32)
        a=torch.tensor(np.stack(actions),dtype=torch.long); r=torch.tensor(rewards,dtype=torch.float32); done=torch.tensor(dones,dtype=torch.float32)
        s,ns=o.flatten(1),no.flatten(1)
        if self.q_based:
            allq=self.local(o); chosen=allq.gather(-1,a[...,None]).squeeze(-1)
            with torch.no_grad(): nextq=self.target_local(no).max(-1).values
            if self.name=='IQL':
                target=r[:,None]+self.gamma*(1-done[:,None])*nextq
                loss=nn.functional.mse_loss(chosen,target)
            elif self.name in ['VDN','QMIX']:
                total=chosen.sum(1) if self.name=='VDN' else self.mixer(chosen,s)
                with torch.no_grad():
                    future=nextq.sum(1) if self.name=='VDN' else self.target_mixer(nextq,ns)
                    target=r+self.gamma*(1-done)*future
                loss=nn.functional.mse_loss(total,target)
            else:
                enc=nn.functional.one_hot(a,5).flatten(1).float()
                joint=self.joint(torch.cat([s,enc],1)).squeeze(1)
                greedy=allq.argmax(-1)
                encgreedy=nn.functional.one_hot(greedy,5).flatten(1).float()
                with torch.no_grad():
                    na=self.target_local(no).argmax(-1)
                    future=self.target_joint(torch.cat([ns,nn.functional.one_hot(na,5).flatten(1).float()],1)).squeeze(1)
                    td=r+self.gamma*(1-done)*future
                    optimal_joint=self.joint(torch.cat([s,encgreedy],1)).squeeze(1)
                offset=self.offset(s).squeeze(1)
                optimal=allq.max(-1).values.sum(1)-optimal_joint+offset
                nonoptimal=torch.minimum(chosen.sum(1)-joint.detach()+offset,torch.zeros_like(r))
                loss=nn.functional.mse_loss(joint,td)+optimal.square().mean()+nonoptimal.square().mean()
            self.optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.parameters(),5); self.optimizer.step()
            self.updates+=1
            if self.updates%50==0:
                self.target_local.load_state_dict(self.local.state_dict())
                if self.name=='QMIX': self.target_mixer.load_state_dict(self.mixer.state_dict())
                if self.name=='QTRAN': self.target_joint.load_state_dict(self.joint.state_dict())
            return float(loss.detach())
        central=self.name in ['MAPPO','PPO']
        with torch.no_grad():
            if self.name=='COMA':
                nd=self.logits(no).softmax(-1); na=nd.argmax(-1)
                bootstrap=(self.coma_q(no[-1:],na[-1:])*nd[-1:]).sum(-1).mean()
            else: bootstrap=self.value(ns[-1:] if central else no[-1:]).mean()
            returns=[]; acc=bootstrap
            for rr,dd in zip(reversed(r),reversed(done)):
                acc=rr+self.gamma*(1-dd)*acc; returns.append(acc)
            returns=torch.stack(list(reversed(returns)))
        old=torch.tensor(np.stack(old_logp),dtype=torch.float32)
        epochs=4 if self.name in ['IPPO','MAPPO','PPO'] else 1
        for _ in range(epochs):
            logits=self.logits(o); dist=Categorical(logits=logits); logp=dist.log_prob(a)
            if self.name=='COMA':
                q=self.coma_q(o,a); taken=q.gather(-1,a[...,None]).squeeze(-1)
                advantage=(taken-(dist.probs*q).sum(-1)).detach()
                value_loss=nn.functional.mse_loss(taken,returns[:,None].expand_as(taken))
            else:
                value=self.value(s if central else o).squeeze(-1)
                value=value[:,None].expand(-1,self.n) if central else value
                advantage=(returns[:,None]-value).detach()
                value_loss=nn.functional.mse_loss(value,returns[:,None].expand_as(value))
            if self.name in ['IPPO','MAPPO','PPO']:
                ratio=(logp-old).exp()
                if self.name=='PPO':
                    ratio=(logp-old).sum(-1,keepdim=True).exp(); advantage=advantage.mean(-1,keepdim=True)
                actor=-torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage).mean()
            else: actor=-(logp*advantage).mean()
            loss=actor+.5*value_loss-.01*dist.entropy().mean()
            self.optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.parameters(),5); self.optimizer.step()
        return float(loss.detach())
