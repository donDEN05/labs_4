"""Prespecified pairwise comparisons; reports consume the saved results."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from shared.data import ROOT
from shared.metrics import interval,paired,save
from lab01_llm.run import sample,frequency,validate_records
from shared.protocol import analysis_inputs

# LLM choice is made on validation, never by maximizing test differences.
choice=json.loads((ROOT/'lab01_llm/results/selection.json').read_text(encoding='utf-8'))
raw=pd.read_json(ROOT/'lab01_llm/results/predictions.jsonl',lines=True)
validate_records(raw.to_dict('records'),require_all=True)
selected=raw[(raw.model==choice['model'])&(raw['mode']==choice['mode'])&(raw.instruction==choice['instruction'])&(raw.split=='test')&(raw.cohort=='main')].set_index('id')
frame=sample();frame=frame[(frame.split=='test')&(frame.cohort=='main')].set_index('id').loc[selected.index]
base=frequency(frame);p=selected.probability.to_numpy(float)
result=paired(frame.label.to_numpy(),np.column_stack([1-p,p]),np.column_stack([1-base,base]),frame.household.to_numpy())
result['evaluated_models']=int(raw.model.nunique())
result['inputs']=analysis_inputs(ROOT/'lab01_llm/results','predictions.jsonl')
result['selection']=choice
result['confirmed_local']=bool(result['ci'][0]>0 and result['p']<.05)
save(ROOT/'lab01_llm/results/paired_baseline.json',result)

episodes=pd.read_csv(ROOT/'lab03_marl/results/episodes.csv')
choice=json.loads((ROOT/'lab03_marl/results/selection.json').read_text(encoding='utf-8'))['algorithm']
d=episodes.query("split == 'test' and regime == 'normal' and ablation == 'none'")
a=d[d.algorithm==choice].pivot(index='seed',columns='episode',values='profit').to_numpy()
b=d[d.algorithm=='Uniform'].pivot(index='seed',columns='episode',values='profit').to_numpy()
assert a.shape==b.shape==(5,24)
delta=a-b;rng=np.random.default_rng(11)
draws=[delta[np.ix_(rng.integers(5,size=5),rng.integers(24,size=24))].mean() for _ in range(1000)]
ci=np.quantile(draws,[.025,.975]).tolist()
save(ROOT/'lab03_marl/results/paired_uniform.json',dict(inputs=analysis_inputs(ROOT/'lab03_marl/results','episodes.csv'),algorithm=choice,mean_difference=float(delta.mean()),ci=ci,bootstrap='paired two-way seed/scenario resampling',seeds=5,scenarios=24,simulation_only=True,confirmed_local=bool(ci[0]>0)))

transfer=pd.read_csv(ROOT/'lab04_optimization/results/transfer.csv')
choice=json.loads((ROOT/'lab04_optimization/results/selection.json').read_text(encoding='utf-8'))['optimizer']
control=next(x for x in transfer.optimizer.unique() if x!=choice)
a=transfer[transfer.optimizer==choice].sort_values('seed').macro_f1.to_numpy();b=transfer[transfer.optimizer==control].sort_values('seed').macro_f1.to_numpy()
assert len(a)==len(b)==5
ci=interval(a-b,11)
save(ROOT/'lab04_optimization/results/paired_transfer.json',dict(inputs=analysis_inputs(ROOT/'lab04_optimization/results','transfer.csv'),optimizer=choice,control=control,mean_difference=float((a-b).mean()),ci=ci,seeds=5,interpretation='Training-seed uncertainty on one fixed test only',confirmed_local=bool(ci[0]>0)))
print('PASS analysis: matched objects/scenarios/seeds; three prespecified comparisons saved')
