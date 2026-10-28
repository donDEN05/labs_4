"""Eight MARL algorithms + uniform allocation + centralized PPO; offline only."""
from collections import deque
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from lab02_multimodal.run import arrays,Model,infer
from lab03_marl.env import DiscountEnv
from lab03_marl.algorithms import Agent
from shared.metrics import interval,save
from shared.data import DATA
from shared.protocol import digest, lock, journal

HERE=Path(__file__).resolve().parent
CFG=json.loads((HERE/'config.json').read_text(encoding='utf-8'))


def features():
    out=HERE/'results';out.mkdir(exist_ok=True)
    scheme=json.loads((HERE.parent/'lab02_multimodal/results/selection.json').read_text(encoding='utf-8'))['scheme']
    lock(out/'features_protocol.json',dict(data=digest(DATA/'panel.jsonl'),forecast=digest(HERE.parent/f'lab02_multimodal/results/{scheme}_11.pt'),scheme=scheme))
    cached=out/'features.npz'
    if cached.exists(): return dict(np.load(cached))
    frame,x,y=arrays()
    scheme=json.loads((HERE.parent/'lab02_multimodal/results/selection.json').read_text(encoding='utf-8'))['scheme']
    model=Model(x[0].shape[1],scheme)
    model.load_state_dict(torch.load(HERE.parent/f'lab02_multimodal/results/{scheme}_11.pt',weights_only=True))
    probability=infer(model,x,np.arange(len(frame)))[:,1]
    frame['forecast']=probability
    cats=json.loads((DATA/'audit.json').read_text(encoding='utf-8'))['categories']
    result={}
    for key,split,cohort in [('train','train','main'),('validation','validation','main'),('test','test','main'),('transfer','test','transfer')]:
        sub=frame[(frame.split==split)&(frame.cohort==cohort)]
        people=[]
        for _,g in sub.groupby('household',sort=True):
            p=g.pivot(index='week',columns='category',values='forecast').reindex(columns=cats)
            people.append(p.to_numpy(np.float32))
        result[key]=np.stack(people)
    tx=pd.read_csv(DATA/'transactions_selected.csv.gz')
    tx=tx[tx.week<=frame[frame.split=='train'].week.max()]
    price=tx.groupby('category')[['sales_value','retail_disc','quantity']].sum()
    result['prices']=((price.sales_value+price.retail_disc)/price.quantity).reindex(cats).to_numpy(float)
    np.savez_compressed(cached,**result)
    save(out/'environment.json',dict(config=CFG,categories=cats,prices=result['prices'].tolist(),forecast_scheme=scheme,forecast_seed=11,source='real historical profiles; simulated discount response',n={k:list(v.shape) for k,v in result.items()}))
    return result


def environment(data,split,regime='normal',personalization=True):
    f=data[split]
    if not personalization:
        f=np.broadcast_to(data['train'].mean(axis=(0,1)),f.shape).copy()
    return DiscountEnv(f,data['prices'],budget=CFG['episode_budget'],cost_ratio=CFG['cost_ratio'],minimum_margin=CFG['minimum_margin_ratio'],elasticity=CFG['elasticity'],substitution=CFG['substitution'],regime=regime)


def evaluate(agent,env,episodes,training_seed,algorithm,split,regime='normal',ablation='none'):
    rows=[];violations=[]
    for ep in range(episodes):
        obs=env.reset(100000+ep);totals=dict(profit=0.,revenue=0.,discount_cost=0.,turnover=0.,proposed_violations=0,applied_violations=0)
        started=time.perf_counter();done=False
        while not done:
            seen=obs.copy()
            if ablation=='no_budget_signal':seen[:,1]=0
            if agent is None:
                # Largest common discount affordable for all categories, using a per-remaining-week budget.
                allowed=env.remaining/max(env.features.shape[1]-env.t,1)/env.prices.sum()
                action=np.full(env.n,min(4,int(allowed/.05)))
            else: action,_=agent.act(seen,explore=False)
            obs,_,done,info=env.step(action)
            for k in totals:totals[k]+=info[k]
            if info['proposed_violations'] or info['applied_violations']:
                violations.append(dict(algorithm=algorithm,seed=training_seed,split=split,regime=regime,ablation=ablation,episode=ep,step=env.t,**info))
        rows.append(dict(algorithm=algorithm,seed=training_seed,split=split,regime=regime,ablation=ablation,episode=ep,elapsed=time.perf_counter()-started,status='complete',**totals))
    return rows,violations


def choose(validation):
    scores=validation.groupby('algorithm').agg(mean_profit=('profit','mean'),sd_profit=('profit','std'),violations=('violations','sum')).reset_index()
    intervals={name:interval(g.profit.to_numpy(),11) for name,g in validation.groupby('algorithm')}
    scores['ci_low']=scores.algorithm.map(lambda name:intervals[name][0])
    scores['ci_high']=scores.algorithm.map(lambda name:intervals[name][1])
    eligible=scores[scores.violations==0]
    best=eligible.sort_values('mean_profit',ascending=False).iloc[0]
    overlap=eligible[(eligible.ci_high>=best.ci_low)&(eligible.ci_low<=best.ci_high)]
    return overlap.sort_values(['sd_profit','mean_profit'],ascending=[True,False]).iloc[0].algorithm,scores


def run(retrain=False):
    torch.set_num_threads(CFG['threads'])
    data=features();out=HERE/'results';curves=[];training=[]
    protocol=dict(config=CFG,features=digest(out/'features.npz'),version=1)
    lock(out/'training_protocol.json',protocol)
    names=CFG['algorithms']+['PPO']
    for name in names:
        for seed in CFG['seeds']:
            checkpoint=out/f'{name}_{seed}.pt';meta=out/f'{name}_{seed}.json'
            state=out/f'{name}_{seed}.journal.json'
            reusable=not state.exists() or json.loads(state.read_text(encoding='utf-8')).get('status')=='complete'
            if checkpoint.exists() and meta.exists() and json.loads(meta.read_text(encoding='utf-8')).get('status')=='complete' and reusable and not retrain:training.append(json.loads(meta.read_text(encoding='utf-8')));continue
            with journal(out/f'{name}_{seed}.journal.json',dict(**protocol,algorithm=name,seed=seed)) as record:
                np.random.seed(seed);torch.manual_seed(seed)
                env=environment(data,'train');obs=env.reset(seed)
                agent=Agent(name,env.n,obs.shape[1],lr=CFG['learning_rate'],gamma=CFG['gamma'])
                replay=deque(maxlen=10000);rollout=[];episode=0;profit=0.;started=time.perf_counter();loss=0.
                rng=np.random.default_rng(seed)
                for step in range(CFG['steps']):
                    action,logp=agent.act(obs,epsilon=max(.05,1-step/CFG['steps']))
                    nxt,reward,done,info=env.step(action)
                    transition=(obs,action,reward,nxt,done,logp)
                    if agent.q_based:
                        replay.append(transition)
                        if len(replay)>=CFG['batch_size'] and step%4==0:
                            idx=rng.choice(len(replay),CFG['batch_size'],replace=False)
                            loss=agent.update([replay[int(i)] for i in idx])
                    else:rollout.append(transition)
                    profit+=info['profit'];obs=nxt
                    if done or step==CFG['steps']-1:
                        if rollout:loss=agent.update(rollout);rollout=[]
                        curves.append(dict(algorithm=name,seed=seed,step=step+1,episode=episode,profit=profit,loss=loss,elapsed=time.perf_counter()-started))
                        profit=0.;episode+=1;obs=env.reset(seed+episode)
                r=dict(algorithm=name,seed=seed,steps=CFG['steps'],elapsed=time.perf_counter()-started,status='complete',parameters=sum(p.numel() for p in agent.parameters() if p.requires_grad),simulation_only=True)
                training.append(r);torch.save(agent.state_dict(),checkpoint);save(meta,r)
                print('TRAIN',name,seed,round(r['elapsed'],2),flush=True)
                record['result']=r
    if curves:pd.DataFrame(curves).to_csv(out/'learning.csv',index=False)
    pd.DataFrame(training).to_csv(out/'training.csv',index=False)
    rows=[];violations=[]
    # Complete validation and freeze winner before the test/stress pass.
    for name in names+['Uniform']:
        for seed in CFG['seeds']:
            env=environment(data,'validation')
            agent=None if name=='Uniform' else Agent(name,env.n,env.observe().shape[1])
            if agent is not None:agent.load_state_dict(torch.load(out/f'{name}_{seed}.pt',weights_only=True))
            r,v=evaluate(agent,env,CFG['validation_episodes'],seed,name,'validation');rows.extend(r);violations.extend(v)
    validation=pd.DataFrame(rows);validation.to_csv(out/'validation.csv',index=False)
    v=validation.groupby(['algorithm','seed']).agg(profit=('profit','mean'),violations=('applied_violations','sum')).reset_index()
    winner,scores=choose(v[v.algorithm.isin(CFG['algorithms'])])
    scores.to_csv(out/'validation_selection.csv',index=False)
    save(out/'selection.json',dict(algorithm=winner,selected_on='validation',rule='zero applied violations, mean profit CI overlap then lower SD',simulation_only=True))
    rows=[]
    for name in names+['Uniform']:
        for seed in CFG['seeds']:
            for split in ['test','transfer']:
                env=environment(data,split);agent=None if name=='Uniform' else Agent(name,env.n,env.observe().shape[1])
                if agent is not None:agent.load_state_dict(torch.load(out/f'{name}_{seed}.pt',weights_only=True))
                r,vv=evaluate(agent,env,CFG['test_episodes'],seed,name,split);rows.extend(r);violations.extend(vv)
            if name==winner:
                for regime in CFG['stress']:
                    r,vv=evaluate(agent,environment(data,'test',regime),CFG['test_episodes'],seed,name,'test',regime=regime);rows.extend(r);violations.extend(vv)
                for ablation in ['no_personalization','no_budget_signal','no_cooperation']:
                    env=environment(data,'test',personalization=ablation!='no_personalization')
                    ablated=agent
                    if ablation=='no_cooperation':
                        ablated=Agent('IQL',env.n,env.observe().shape[1]);ablated.load_state_dict(torch.load(out/f'IQL_{seed}.pt',weights_only=True))
                    r,vv=evaluate(ablated,env,CFG['test_episodes'],seed,name,'test',ablation=ablation);rows.extend(r);violations.extend(vv)
    episodes=pd.DataFrame(rows);episodes.to_csv(out/'episodes.csv',index=False)
    pd.DataFrame(violations).to_json(out/'violations.jsonl',orient='records',lines=True)
    standard=episodes.query("split == 'test' and regime == 'normal' and ablation == 'none'")
    summary=[]
    for name,g in standard.groupby('algorithm'):
        seeds=g.groupby('seed').profit.mean()
        low,high=interval(seeds.to_numpy(),11)
        summary.append(dict(algorithm=name,profit_mean=float(seeds.mean()),profit_sd=float(seeds.std()),ci_low=low,ci_high=high,discount_cost=float(g.discount_cost.mean()),turnover=float(g.turnover.mean()),proposed_violations=int(g.proposed_violations.sum()),applied_violations=int(g.applied_violations.sum()),elapsed=float(g.elapsed.mean()),n_seeds=len(seeds)))
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    save(out/'risk_card.json',dict(real_world_causal_effect='not identified',cost_ratio='assumed',response_elasticity='assumed',unit_demand='Bernoulli one unit per category/week',constraints='hard budget and margin controller',selection=winner,limitations=['short budget, no convergence claim','grocery-to-ecommerce domain shift','greedy budget projection can favor category order','income/demographic attributes excluded','no online deployment']))
    print('SELECTED',winner,flush=True)


if __name__=='__main__':run(retrain='--retrain' in sys.argv)
