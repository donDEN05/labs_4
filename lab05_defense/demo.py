"""Replay a real saved LLM prediction, fuse it with lab02 and show risk-controlled actions."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from shared.data import load
from shared.metrics import save
from lab03_marl.run import features,environment,CFG
from lab03_marl.algorithms import Agent


def demo():
    root=Path(__file__).resolve().parents[1]
    selected=json.loads((root/'lab01_llm/results/selection.json').read_text(encoding='utf-8'))
    predictions=pd.read_json(root/'lab01_llm/results/predictions.jsonl',lines=True)
    chosen=predictions[(predictions.model==selected['model'])&(predictions['mode']==selected['mode'])&(predictions.instruction==selected['instruction'])&(predictions.split=='test')&(predictions.cohort=='main')].iloc[0]
    panel=load();profile=panel[panel.id==chosen.id].iloc[0]
    torch.set_num_threads(2)
    data=features();env=environment(data,'test')
    households=sorted(panel[(panel.split=='test')&(panel.cohort=='main')].household.unique())
    weeks=sorted(panel[(panel.split=='test')&(panel.cohort=='main')].week.unique())
    env.reset(11);env.person=households.index(profile.household);env.t=weeks.index(profile.week)
    obs=env.observe()
    categories=json.loads((root/'shared/data/audit.json').read_text(encoding='utf-8'))['categories']
    category=categories.index(profile.category)
    numerical_p=float(obs[category,0]);llm_p=float(chosen.probability)
    obs[category,0]=(numerical_p+llm_p)/2
    # This blend demonstrates interfaces; its joint policy quality has not been separately validated.
    algorithm=json.loads((root/'lab03_marl/results/selection.json').read_text(encoding='utf-8'))['algorithm']
    agent=Agent(algorithm,env.n,obs.shape[1]);agent.load_state_dict(torch.load(root/f'lab03_marl/results/{algorithm}_11.pt',weights_only=True))
    actions,_=agent.act(obs,explore=False)
    _,reward,done,info=env.step(actions)
    result=dict(mode='offline replay of saved real LLM inference + live local model/policy',id=profile.id,category=profile.category,week=int(profile.week),history_end=int(profile.history_end),llm=selected,llm_probability=llm_p,multimodal_probability=numerical_p,blend_probability=float(obs[category,0]),algorithm=algorithm,categories=categories,proposal_percent=(env.grid[actions]*100).tolist(),applied_percent=(np.array(info['discounts'])*100).tolist(),simulation=info,live_customer_offers_sent=0,limitation='The blend is an integration demonstration, not a validated joint causal policy')
    save(root/'lab05_defense/results/demo.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':demo()
