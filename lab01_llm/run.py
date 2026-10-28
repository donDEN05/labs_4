"""Eight real LLMs, two prompts, zero/six-shot; no API or fake model outputs."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from shared.data import ROOT, load
from shared.metrics import classification, f1_interval, save
from shared.protocol import digest, fingerprint, lock

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))


def parse_prediction(raw):
    try:
        obj = json.loads(raw.strip())
        p = obj['p']
        if set(obj) != {'p'} or type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError('invalid schema/probability')
        return float(p)
    except (TypeError, KeyError, json.JSONDecodeError) as e:
        raise ValueError('invalid JSON prediction') from e


def summarize(frame):
    # Invalid answers count as wrong in classification; probability calibration is also reported valid-only.
    y = frame.label.to_numpy(int)
    valid = frame.valid.to_numpy(bool)
    prob = frame.probability.to_numpy(float)
    scored = np.where(valid, prob, 1 - y)
    r = classification(y, np.column_stack([1 - scored, scored]))
    r.update(invalid_rate=float(1 - valid.mean()), latency_s=float(frame.elapsed.mean()))
    if valid.any():
        r['valid_only'] = classification(y[valid], np.column_stack([1 - prob[valid], prob[valid]]))
    return r


def sample():
    frame = load()
    parts = []
    for split, cohort, n in [('validation','main',CONFIG['validation_n']), ('test','main',CONFIG['test_n']), ('test','transfer',CONFIG['transfer_n'])]:
        sub = frame[(frame.split == split) & (frame.cohort == cohort)].copy()
        # Label-blind hash sampling before seeing any model's outputs.
        sub['order'] = sub.id.map(lambda x:hashlib.sha256((str(CONFIG['seed']) + x).encode()).hexdigest())
        parts.append(sub.sort_values('order').head(n).drop(columns='order'))
    return pd.concat(parts).reset_index(drop=True)


def prompt(text, variant):
    instruction = [
        'Forecast whether this household will buy in this category next week using only the past observations. Return only JSON {"p":number}, where p is the probability from 0 to 1 of at least one purchase. No explanation.',
        'Estimate next-week purchase probability for the category from the four historical weeks. Output exactly one JSON object with the sole numeric key p between 0 and 1. Do not add prose.'
    ][variant]
    return instruction + '\n' + text


def download():
    import subprocess
    subprocess.run([sys.executable,str(ROOT/'shared/download_models.py')],check=True)


def model_manifest():
    result={}
    for path in (HERE/'results').glob('model_manifest*.json'):
        for name,value in json.loads(path.read_text(encoding='utf-8')).items():
            if name in result and result[name]['revision']!=value['revision']:raise ValueError('conflicting revisions')
            result[name]=value
    return result


def protocol():
    return dict(config=CONFIG,data_sha256=digest(ROOT/'shared/data/panel.jsonl'),decoder='INT8 linear / float32 embeddings; binary digit logits; prompt v1',models={k:v['revision'] for k,v in model_manifest().items()})


def validate_records(records,require_all=False):
    selected=sample().set_index('id'); manifest=model_manifest(); signature=fingerprint(protocol())
    expected={(name,i,mode,v) for name in CONFIG['models'] for i in selected.index for mode in ('zero-shot','six-shot') for v in (0,1)}
    seen=set()
    for r in records:
        key=(r['model'],r['id'],r['mode'],r['instruction'])
        if key not in expected or key in seen:raise ValueError('unexpected or duplicate prediction identity')
        row=selected.loc[r['id']]
        if r['status']!='complete' or not r['valid'] or r['revision']!=manifest[r['model']]['revision'] or r.get('protocol_hash')!=signature or any(r[k]!=row[k] for k in ('label','split','cohort','household','category','week')) or not math.isfinite(r['probability']) or not 0<=r['probability']<=1:
            raise ValueError('failed, stale or mismatched prediction')
        seen.add(key)
    if require_all and seen!=expected:raise ValueError(f'LLM experiment incomplete: {len(seen)}/{len(expected)}')
    return {name for name in CONFIG['models'] if {k for k in expected if k[0]==name}<=seen}


def binary_probability(logits, zero, one):
    scores=np.asarray(logits,dtype=float)[:,[zero,one]]
    scores=np.exp(scores-scores.max(1,keepdims=True))
    return scores[:,1]/scores.sum(1)


def frequency(frame):
    return ((np.stack(frame.numbers)[:,:4]>0).sum(1)+1)/6


def quantize(model):
    import torch
    # Traverse lazily: a list would retain every old float layer until the end.
    for parent in model.modules():
        for name,child in list(parent.named_children()):
            if isinstance(child,torch.nn.Linear):
                child=child.float();child.qconfig=torch.ao.quantization.default_dynamic_qconfig
                setattr(parent,name,torch.ao.nn.quantized.dynamic.Linear.from_float(child))
    return model.float()


def run(names=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(CONFIG['threads']); torch.manual_seed(CONFIG['seed'])
    out=HERE/'results'; out.mkdir(exist_ok=True)
    selected=sample(); selected.to_json(out/'sample.jsonl',orient='records',lines=True,force_ascii=False)
    train=load().query("split == 'train' and cohort == 'main'")
    examples=pd.concat([train[train.label==y].sample(3,random_state=11) for y in (0,1)]).sample(frac=1,random_state=11)
    save(out/'fewshot_ids.json',examples.id.tolist())
    manifest=model_manifest()
    signature=lock(out/'experiment_protocol.json',protocol())
    path=out/'predictions.jsonl'
    existing=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    completed=validate_records(existing)
    done={(r['model'],r['id'],r['mode'],r['instruction']) for r in existing}
    probes_path=out/'free_json_probes.jsonl'
    probes=[json.loads(s) for s in probes_path.read_text(encoding='utf-8').splitlines()] if probes_path.exists() else []
    probed={(r['model'],r['mode'],r['instruction']) for r in probes}
    instructions=[
        'Forecast whether this household buys in the category next week. Return JSON {"label":0} for no purchase or {"label":1} for a purchase. Use only past observations.',
        'Predict a category purchase in the next week from the historical profile. Output exactly {"label":1} if a purchase is likely, otherwise {"label":0}.'
    ]
    for name in names or CONFIG['models']:
        if name in completed and all((name,mode,v) in probed for mode in ('zero-shot','six-shot') for v in (0,1)):continue
        revision=manifest[name]['revision']; source=ROOT/'shared/models'/name.replace('/','--')
        print('LOAD',name,flush=True)
        tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True,padding_side='left')
        tokenizer.pad_token=tokenizer.eos_token
        model=AutoModelForCausalLM.from_pretrained(source,local_files_only=True,dtype=torch.bfloat16).eval()
        # Quantize one linear layer at a time: a 3B model never needs a full float32 copy in 16 GB RAM.
        model=quantize(model)
        size=sum(p.numel()*p.element_size() for p in model.parameters())
        size+=sum(m.weight().numel()*m.weight().element_size()+(m.bias().numel()*4 if m.bias() is not None else 0) for m in model.modules() if isinstance(m,torch.ao.nn.quantized.dynamic.Linear))
        zero,one=tokenizer.encode('0',add_special_tokens=False),tokenizer.encode('1',add_special_tokens=False)
        if len(zero)!=1 or len(one)!=1: raise ValueError('binary digits must be single tokens')
        for mode in ['zero-shot','six-shot']:
            for variant in (0,1):
                pending=[r for r in selected.itertuples() if (name,r.id,mode,variant) not in done]
                for start in range(0,len(pending),CONFIG['batch_size']):
                    batch=pending[start:start+CONFIG['batch_size']]; rendered=[]
                    for row in batch:
                        messages=[]
                        if mode=='six-shot':
                            for e in examples.itertuples():
                                messages.extend([{'role':'user','content':e.text},{'role':'assistant','content':json.dumps({'label':int(e.label)})}])
                        messages.append({'role':'user','content':instructions[variant]+'\n'+row.text})
                        rendered.append(tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)+'{"label":')
                    inputs=tokenizer(rendered,return_tensors='pt',padding=True,truncation=False)
                    if inputs.input_ids.shape[1]>CONFIG['max_input_tokens']: raise ValueError('context exceeds fixed budget')
                    t=time.perf_counter()
                    with torch.inference_mode(): logits=model(**inputs,use_cache=False,logits_to_keep=1).logits[:,-1].float().numpy()
                    elapsed=(time.perf_counter()-t)/len(batch)
                    probs=binary_probability(logits,zero[0],one[0])
                    for j,(row,p) in enumerate(zip(batch,probs)):
                        label=int(p>.5); raw=json.dumps({'label':label})
                        result=dict(id=row.id,household=row.household,category=row.category,week=int(row.week),split=row.split,cohort=row.cohort,label=int(row.label),history_length=len(row.text),model=name,revision=revision,mode=mode,instruction=variant,raw=raw,probability=float(p),valid=True,elapsed=elapsed,model_bytes=size,input_tokens=int(inputs.attention_mask[j].sum()),output_tokens=1,status='complete',decoder='constrained binary JSON',digit_logits=[float(logits[j,zero[0]]),float(logits[j,one[0]])],unconstrained_token_valid=int(logits[j].argmax()) in (zero[0],one[0]))
                        result['protocol_hash']=signature
                        with path.open('a',encoding='utf-8') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
                        done.add((name,row.id,mode,variant))
                    print(name,mode,variant,start,'batch_s',round(elapsed*len(batch),2),flush=True)
                # One fixed test profile per condition: free generation compliance is a separate small probe.
                if (name,mode,variant) in probed:continue
                row=selected[(selected.split=='test')&(selected.cohort=='main')].iloc[0]
                probe=prompt(row.text,variant)
                messages=[]
                if mode=='six-shot':
                    for e in examples.itertuples():messages.extend([{'role':'user','content':e.text},{'role':'assistant','content':json.dumps({'p':float(e.label)})}])
                messages.append({'role':'user','content':probe})
                text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
                inputs=tokenizer(text,return_tensors='pt')
                t=time.perf_counter()
                with torch.inference_mode():tokens=model.generate(**inputs,max_new_tokens=CONFIG['max_new_tokens'],do_sample=False,pad_token_id=tokenizer.eos_token_id)
                raw=tokenizer.decode(tokens[0,inputs.input_ids.shape[1]:],skip_special_tokens=True)
                try:parse_prediction(raw); valid=True
                except ValueError:valid=False
                with probes_path.open('a',encoding='utf-8') as f:f.write(json.dumps(dict(model=name,mode=mode,instruction=variant,id=row.id,raw=raw,valid=valid,elapsed=time.perf_counter()-t))+'\n')
        del model,tokenizer;gc.collect();aggregate()
    aggregate()


def aggregate():
    path=HERE/'results/predictions.jsonl'
    if not path.exists(): return
    frame=pd.read_json(path,lines=True)
    complete=validate_records(frame.to_dict('records'))
    frame=frame[frame.model.isin(complete)]
    if frame.empty:return
    selected=sample(); bp=frequency(selected)
    baseline=[]
    for (split,cohort),g in selected.assign(p=bp).groupby(['split','cohort']):
        baseline.append(dict(split=split,cohort=cohort,**classification(g.label.to_numpy(),np.column_stack([1-g.p,g.p]))))
    pd.DataFrame(baseline).to_csv(HERE/'results/baseline.csv',index=False)
    rows=[]
    for key,g in frame.groupby(['model','mode','instruction','split','cohort']):
        r=summarize(g)
        y=g.label.to_numpy(int); p=np.where(g.valid,g.probability,1-y).astype(float)
        ci=f1_interval(y,np.column_stack([1-p,p]),g.household, repeats=1000)
        rows.append(dict(zip(['model','mode','instruction','split','cohort'],key)) | {k:v for k,v in r.items() if k not in ('confusion','valid_only')} | {'ci_low':ci[0],'ci_high':ci[1]})
    metrics=pd.DataFrame(rows)
    metrics.to_csv(HERE/'results/metrics.csv',index=False)
    v=metrics[(metrics.split=='validation') & (metrics.cohort=='main') & (metrics.n==CONFIG['validation_n'])]
    eligible=v[v.invalid_rate<=.05]
    if len(eligible):
        best=eligible[eligible.macro_f1>=eligible.macro_f1.max()-.01].sort_values(['brier','latency_s']).iloc[0]
        selection=dict(status='selected',model=best.model,mode=best['mode'],instruction=int(best.instruction),rule='validation invalid<=5%, F1 within .01 then Brier then latency')
    else:
        selection=dict(status='no_eligible_model',reason='all evaluated configurations exceed invalid answer threshold')
    selection['evaluated_models']=int(frame.model.nunique())
    save(HERE/'results/selection.json',selection)
    frame[(~frame.valid) | ((frame.probability>.5)!=frame.label)].to_csv(HERE/'results/errors.csv',index=False)
    analysis=frame.assign(error=(frame.probability>.5)!=frame.label,language='en',length_group=pd.cut(frame.history_length,[0,150,220,10000],labels=['short','medium','long']))
    summaries=[]
    for dimension in ['label','category','language','length_group']:
        g=analysis[analysis.split=='test'].groupby(['model','mode','instruction',dimension],observed=True).agg(n=('id','size'),error_rate=('error','mean')).reset_index()
        g=g.rename(columns={dimension:'group'});g['dimension']=dimension;summaries.append(g)
    pd.concat(summaries).to_csv(HERE/'results/error_groups.csv',index=False)
    stability=[]
    for (model,mode),g in frame[(frame.split=='test')&(frame.cohort=='main')].groupby(['model','mode']):
        a=g[g.instruction==0].set_index('id');b=g[g.instruction==1].set_index('id');ids=a.index.intersection(b.index)
        if len(ids):
            aa=a.loc[ids].probability.to_numpy(float);bb=b.loc[ids].probability.to_numpy(float)
            stability.append(dict(model=model,mode=mode,n=len(ids),instruction_flip_rate=float(((aa>.5)!=(bb>.5)).mean()),mean_probability_change=float(np.abs(aa-bb).mean())))
    pd.DataFrame(stability).to_csv(HERE/'results/stability.csv',index=False)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--download',action='store_true'); ap.add_argument('--model',action='append'); ap.add_argument('--aggregate',action='store_true')
    args=ap.parse_args()
    if args.download: download()
    elif args.aggregate: aggregate()
    else: run(args.model)
