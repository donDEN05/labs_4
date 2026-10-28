"""Catches overspending, missing cooperation and no-op algorithm updates."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from lab03_marl.env import DiscountEnv
from lab03_marl.algorithms import Agent
from lab03_marl.run import choose
import pandas as pd

features=np.full((3,4,8),.3,dtype=np.float32)
for kw in [dict(cost_ratio=1.),dict(cost_ratio=float('nan')),dict(minimum_margin=-.1),dict(cost_ratio=.8,minimum_margin=.2,regime='high_cost')]:
    try:
        DiscountEnv(features,np.ones(8),**kw)
        raise AssertionError('infeasible margin accepted')
    except ValueError:pass
for prices,budget in [(np.ones(8),-1),(np.zeros(8),5)]:
    try:
        DiscountEnv(features,prices=prices,budget=budget)
        raise AssertionError('invalid budget/price accepted')
    except ValueError:
        pass
env=DiscountEnv(features,prices=np.ones(8)*10,budget=0)
env.reset(11)
_,_,_,info=env.step(np.full(8,4))
assert info['discount_cost']==0 and info['applied_violations']==0 and info['proposed_violations']>0
env=DiscountEnv(features,prices=np.ones(8)*10,budget=2)
obs=env.reset(11)
_,r,_,a=env.step(np.full(8,4))
assert a['discount_cost']<=2 and a['minimum_margin']>=.1-1e-6
env.reset(11); _,r2,_,b=env.step(np.full(8,4))
assert r==r2 and a['discount_cost']==b['discount_cost']
full=env.project(np.full(8,4),2.)
other=env.project(np.array([4,4,4,4,4,4,4,0]),2.)
assert not np.array_equal(full[:-1],other[:-1]), 'categories do not compete for budget'
stress=DiscountEnv(features,prices=np.ones(8)*10,budget=100,regime='high_cost')
stress.reset(11)
_,_,_,s=stress.step(np.full(8,4))
assert s['minimum_margin']>=.1-1e-6 and s['applied_violations']==0
assert s['proposed_violations']==8, 'stress margin violations before control must be recorded'
try:
    stress.step(np.full(8,5))
    raise AssertionError('out-of-range action accepted')
except ValueError:
    pass
candidate=pd.DataFrame([dict(algorithm='A',seed=i,profit=v,violations=0) for i,v in enumerate([8,9,10,11,12])]+[dict(algorithm='B',seed=i,profit=v,violations=0) for i,v in enumerate([9.8,9.9,9.8,9.9,9.8])])
assert choose(candidate)[0]=='B','overlapping CIs should favor stability'
torch.set_num_threads(2)
for name in ['IQL','VDN','QMIX','IAC','IPPO','MAPPO','COMA','QTRAN','PPO']:
    torch.manual_seed(11)
    agent=Agent(name,8,obs.shape[1])
    before=[p.detach().clone() for p in agent.parameters()]
    transitions=[]
    for i in range(4):
        action,logp=agent.act(obs)
        nxt,reward,done,_=env.step(action)
        transitions.append((obs,action,reward,nxt,done,logp))
        obs=env.reset(12) if done else nxt
    loss=agent.update(transitions)
    assert np.isfinite(loss),name
    assert any(not torch.equal(a,b) for a,b in zip(before,agent.parameters())),name
print('PASS lab03: hard budget/margin, deterministic seed, shared constraint, nine real policy updates')
