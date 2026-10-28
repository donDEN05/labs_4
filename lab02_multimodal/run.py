"""Eight small, actually different fusion schemes on one causal panel."""
import argparse
import copy
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch
from torch import nn
from sklearn.feature_extraction.text import TfidfVectorizer
from shared.data import load
from shared.metrics import classification, f1_interval, paired, save
from shared.protocol import digest, fingerprint, journal, lock

HERE=Path(__file__).resolve().parent
CFG=json.loads((HERE/'config.json').read_text(encoding='utf-8'))


def image_from_history(numbers):
    n=np.asarray(numbers).reshape(3,4)
    channels=[]
    for series in n:
        im=Image.new('L',(16,16),0)
        scaled=series/max(float(series.max()),1.)
        points=[(int(i*5),int(14-13*v)) for i,v in enumerate(scaled)]
        ImageDraw.Draw(im).line(points,fill=255,width=2)
        channels.append(np.asarray(im,dtype=np.float32)/255)
    return np.stack(channels)


class Model(nn.Module):
    # ponytail: small superset model; prune inactive branches when deployment cost matters.
    def __init__(self,text_dim,scheme='all'):
        super().__init__()
        self.scheme=scheme
        self.text=nn.Sequential(nn.Linear(text_dim,16),nn.ReLU())
        self.numbers=nn.Sequential(nn.Linear(12,16),nn.ReLU())
        self.image=nn.Sequential(nn.Conv2d(3,4,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((2,2)),nn.Flatten(),nn.Linear(16,16),nn.ReLU())
        self.early=nn.Sequential(nn.Linear(text_dim+12,32),nn.ReLU(),nn.Linear(32,2))
        self.head=nn.Linear(48,2)
        self.heads=nn.ModuleList([nn.Linear(16,2) for _ in range(3)])
        self.active={'text':[0],'numbers':[1],'image':[2],'early':[0,1],'late':[0,1],'text_image':[0,2],'numbers_image':[1,2],'all':[0,1,2]}[scheme]

    def forward(self,text,numbers,image,mask=None):
        if mask is None: mask=torch.ones(len(text),3,device=text.device)
        active=torch.zeros_like(mask); active[:,self.active]=1
        mask=mask*active
        text,numbers,image=[torch.where(m.bool(),v,0.) for v,m in zip((text,numbers,image),(mask[:,0:1],mask[:,1:2],mask[:,2,None,None,None]))]
        if not all(torch.isfinite(v).all() for v in (text,numbers,image)):
            raise ValueError('present modalities must be finite')
        if self.scheme=='early': return self.early(torch.cat([text,numbers],1))
        features=[self.text(text),self.numbers(numbers),self.image(image)]
        features=[f*mask[:,i:i+1] for i,f in enumerate(features)]
        if self.scheme=='late':
            return sum(self.heads[i](features[i])*mask[:,i:i+1] for i in self.active)/mask[:,self.active].sum(1,keepdim=True).clamp(min=1)
        return self.head(torch.cat(features,1))


def arrays(frame=None):
    frame=load() if frame is None else frame
    out=HERE/'results'; out.mkdir(exist_ok=True)
    vector_path=out/'preprocess.pkl'
    raw=np.stack(frame.numbers).astype(np.float32)
    train=(frame.split=='train') & (frame.cohort=='main')
    if vector_path.exists():
        vector,mean,std=pickle.loads(vector_path.read_bytes())
    else:
        vector=TfidfVectorizer(max_features=CFG['text_features'],ngram_range=(1,2),token_pattern=r'(?u)\b\w+\b')
        vector.fit(frame.loc[train,'text'])
        mean,std=raw[train].mean(0),raw[train].std(0).clip(.001)
        vector_path.write_bytes(pickle.dumps((vector,mean,std)))
    text=vector.transform(frame.text).toarray().astype(np.float32)
    images=np.stack([image_from_history(x) for x in raw])
    return frame,(torch.from_numpy(text),torch.from_numpy((raw-mean)/std),torch.from_numpy(images)),torch.tensor(frame.label.to_numpy(),dtype=torch.long)


def infer(model,x,idx,mask=None):
    model.eval()
    outputs=[]
    with torch.inference_mode():
        for chunk in np.array_split(np.asarray(idx),max(1,int(np.ceil(len(idx)/1024)))):
            if len(chunk): outputs.append(model(*(v[chunk] for v in x),None if mask is None else mask[chunk]).softmax(1).numpy())
    return np.concatenate(outputs)


def fit(scheme,seed,x,y,train_idx,val_idx,steps=None,optimizer='AdamW',lr=None,record_noise=False):
    inputs=hashlib.sha256(b''.join(v.numpy().tobytes() for v in (*x,y))).hexdigest()
    protocol=dict(scheme=scheme,seed=seed,steps=steps or CFG['steps'],optimizer=optimizer,lr=lr or CFG['learning_rate'],batch_size=CFG['batch_size'],inputs=inputs,train=hashlib.sha256(np.asarray(train_idx).tobytes()).hexdigest(),validation=hashlib.sha256(np.asarray(val_idx).tobytes()).hexdigest(),version=1)
    with journal(HERE.parent/'shared/run_journal'/f'{fingerprint(protocol)}.json',protocol) as record:
        result=_fit(scheme,seed,x,y,train_idx,val_idx,steps,optimizer,lr,record_noise)
        result[2]['protocol_hash']=fingerprint(protocol)
        record['result']=result[2]
        return result


def _fit(scheme,seed,x,y,train_idx,val_idx,steps=None,optimizer='AdamW',lr=None,record_noise=False):
    torch.manual_seed(seed)
    model=Model(x[0].shape[1],scheme)
    initial=hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
    kw={'lr':lr or CFG['learning_rate']}
    name=optimizer
    if name=='Momentum': name='SGD'; kw['momentum']=.9
    if name=='Nesterov': name='SGD'; kw.update(momentum=.9,nesterov=True)
    opt=getattr(torch.optim,name)(model.parameters(),**kw)
    rng=np.random.default_rng(seed)
    batches=rng.choice(train_idx,size=(steps or CFG['steps'],CFG['batch_size']),replace=True)
    order=hashlib.sha256(batches.tobytes()).hexdigest()
    best,best_score,curve=None,-1,[]
    started=time.perf_counter()
    for step,batch in enumerate(batches,1):
        model.train(); opt.zero_grad()
        logits=model(*(a[batch] for a in x))
        loss=nn.functional.cross_entropy(logits,y[batch])
        if not torch.isfinite(loss):
            return model,curve,dict(status='unstable',seed=seed,step=step,elapsed=time.perf_counter()-started,initial_state_hash=initial,batch_order_hash=order)
        loss.backward()
        grad_norm=float(torch.sqrt(sum((p.grad**2).sum() for p in model.parameters() if p.grad is not None)))
        nn.utils.clip_grad_norm_(model.parameters(),5.)
        opt.step()
        if step==1 or step%50==0 or step==len(batches):
            p=infer(model,x,val_idx)
            score=classification(y[val_idx].numpy(),p)['macro_f1']
            curve.append(dict(step=step,loss=float(loss.detach()),val_macro_f1=score,grad_norm=grad_norm,elapsed=time.perf_counter()-started))
            if score>best_score:
                best_score=score; best=copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    return model,curve,dict(status='complete',seed=seed,elapsed=time.perf_counter()-started,initial_state_hash=initial,batch_order_hash=order,val_macro_f1=best_score,parameters=sum(p.numel() for p in model.parameters()))


def run():
    torch.set_num_threads(CFG['threads'])
    frame,x,y=arrays()
    lock(HERE/'results/cache_protocol.json',dict(config=CFG,data=digest(HERE.parent/'shared/data/panel.jsonl'),preprocess=digest(HERE/'results/preprocess.pkl')))
    tr=np.flatnonzero((frame.split=='train') & (frame.cohort=='main'))
    va=np.flatnonzero((frame.split=='validation') & (frame.cohort=='main'))
    out=HERE/'results'; rows=[]; curves=[]
    lock(out/'protocol.json',dict(config=CFG,sample_sha256=hashlib.sha256((HERE.parent/'shared/data/panel.jsonl').read_bytes()).hexdigest(),n_train=len(tr),n_validation=len(va),fit_on='train only'))
    # First complete all validation comparisons; freeze selection before test predictions.
    for scheme in CFG['schemes']:
        for seed in CFG['seeds']:
            checkpoint=out/f'{scheme}_{seed}.pt'
            meta=out/f'{scheme}_{seed}.json'
            if meta.exists() and checkpoint.exists() and json.loads(meta.read_text(encoding='utf-8')).get('status')=='complete':
                r=json.loads(meta.read_text(encoding='utf-8')); rows.append(r)
                continue
            model,curve,r=fit(scheme,seed,x,y,tr,va)
            t=time.perf_counter(); p=infer(model,x,va); latency=(time.perf_counter()-t)/len(va)
            r.update(scheme=scheme,latency_s=latency,**classification(y[va].numpy(),p))
            rows.append(r); curves.extend([dict(scheme=scheme,seed=seed,**c) for c in curve])
            torch.save(model.state_dict(),checkpoint); save(meta,r)
            print(scheme,seed,round(r['macro_f1'],4),round(r['elapsed'],2),flush=True)
    pd.DataFrame(rows).to_csv(out/'validation.csv',index=False)
    if len(rows)!=len(CFG['schemes'])*len(CFG['seeds']) or any(r['status']!='complete' for r in rows):raise ValueError('incomplete fusion experiment; inspect shared/run_journal')
    if curves: pd.DataFrame(curves).to_csv(out/'learning.csv',index=False)
    v=pd.DataFrame(rows).query("status == 'complete'").groupby('scheme')[['macro_f1','latency_s']].mean().reset_index()
    selected=v[v.macro_f1>=v.macro_f1.max()-.01].sort_values('latency_s').iloc[0].scheme
    save(out/'selection.json',dict(scheme=selected,rule=CFG['selection'],selected_on='validation',evaluated_schemes=len(v)))
    predictions=[]; metrics=[]; ensembles={}
    for scheme in CFG['schemes']:
        for split,cohort in [('test','main'),('test','transfer')]:
            idx=np.flatnonzero((frame.split==split)&(frame.cohort==cohort)); probs=[]
            for seed in CFG['seeds']:
                model=Model(x[0].shape[1],scheme)
                model.load_state_dict(torch.load(out/f'{scheme}_{seed}.pt',weights_only=True))
                p=infer(model,x,idx); probs.append(p)
                r=classification(y[idx].numpy(),p)
                metrics.append(dict(scheme=scheme,seed=seed,split=split,cohort=cohort,**r))
            p=np.mean(probs,axis=0)
            ensembles[(scheme,cohort)]=p
            ci=f1_interval(y[idx].numpy(),p,frame.iloc[idx].household,repeats=CFG['bootstrap_repeats'])
            metrics.append(dict(scheme=scheme,seed='ensemble',split=split,cohort=cohort,ci_low=ci[0],ci_high=ci[1],**classification(y[idx].numpy(),p)))
            predictions.append(frame.iloc[idx][['id','household','category','week','cohort','label']].assign(scheme=scheme,p=p[:,1]))
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    pd.concat(predictions).to_csv(out/'predictions.csv',index=False)
    idx=np.flatnonzero((frame.split=='test')&(frame.cohort=='main'))
    tests=[]
    # Comparator is frozen by validation, not the largest test score.
    unimodal=v[v.scheme.isin(['text','numbers','image'])].sort_values('macro_f1',ascending=False).iloc[0].scheme
    for scheme in CFG['schemes']:
        tests.append(dict(scheme=scheme,baseline=unimodal,**paired(y[idx].numpy(),ensembles[(scheme,'main')],ensembles[(unimodal,'main')],frame.iloc[idx].household,repeats=1000)))
    tests=pd.DataFrame(tests); tests['p_bonferroni']=(tests.p*len(tests)).clip(upper=1); tests.to_csv(out/'permutation.csv',index=False)
    model=Model(x[0].shape[1],selected); model.load_state_dict(torch.load(out/f'{selected}_11.pt',weights_only=True))
    ablations=[]
    for modality in range(3):
        mask=torch.ones(len(frame),3); mask[:,modality]=0
        p=infer(model,x,idx,mask)
        ablations.append(dict(missing=['text','numbers','image'][modality],scheme=selected,seed=11,**classification(y[idx].numpy(),p)))
    pd.DataFrame(ablations).to_csv(out/'ablations.csv',index=False)
    p=ensembles[(selected,'main')]
    errors=frame.iloc[idx].copy(); errors['p']=p[:,1]
    errors[errors.label!=(p.argmax(1))].drop(columns=['numbers']).to_csv(out/'errors.csv',index=False)
    print('SELECTED',selected,flush=True)


if __name__=='__main__': run()
