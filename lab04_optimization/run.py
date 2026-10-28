"""Controlled optimizer experiment using exactly lab02's model and data."""
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from lab02_multimodal.run import arrays,fit,infer,Model
from shared.metrics import classification,interval,save
from shared.protocol import lock

HERE=Path(__file__).resolve().parent
CFG=json.loads((HERE/'config.json').read_text(encoding='utf-8'))


def successful(frame):
    return frame[frame.status=='complete'].copy()


def resources():
    """Measure tensor storage after a real update; this is not process peak RSS."""
    torch.set_num_threads(CFG['threads'])
    frame,x,y=arrays();batch=np.flatnonzero((frame.split=='train')&(frame.cohort=='main'))[:CFG['batch_size']]
    rows=[]
    for name in CFG['optimizers']:
        model=Model(x[0].shape[1],'all')
        model.load_state_dict(torch.load(HERE/f'results/{name}_11.pt',weights_only=True))
        kw=dict(lr=.001)
        if name in ('Momentum','Nesterov'):kw.update(momentum=.9,nesterov=name=='Nesterov')
        opt=getattr(torch.optim,'SGD' if name in ('Momentum','Nesterov') else name)(model.parameters(),**kw)
        torch.nn.functional.cross_entropy(model(*(v[batch] for v in x)),y[batch]).backward();opt.step()
        size=lambda tensors:sum(v.numel()*v.element_size() for v in tensors)
        parameters=size(model.parameters());gradients=size(p.grad for p in model.parameters() if p.grad is not None)
        states=size(v for state in opt.state.values() for v in state.values() if torch.is_tensor(v))
        for seed in CFG['seeds']:
            assert size(torch.load(HERE/f'results/{name}_{seed}.pt',weights_only=True).values())==parameters
        rows.append(dict(optimizer=name,profile_lr=.001,profile_seed=11,profile_steps=1,parameter_bytes=parameters,gradient_bytes=gradients,optimizer_state_bytes=states,total_tensor_bytes=parameters+gradients+states,scope='one real update on seed-11 checkpoint; storage shapes verified for all five seeds; excludes activations/runtime/RSS'))
    pd.DataFrame(rows).to_csv(HERE/'results/memory.csv',index=False)


def gradient_noise(model,x,y,idx):
    grads=[]
    model.eval()
    for block in np.array_split(idx[:1024],8):
        model.zero_grad(); loss=torch.nn.functional.cross_entropy(model(*(v[block] for v in x)),y[block]); loss.backward()
        grads.append(torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).detach().numpy())
    g=np.stack(grads)
    return dict(trace_covariance=float(g.var(0,ddof=1).sum()),mean_gradient_squared=float(np.square(g.mean(0)).sum()),noise_ratio=float(g.var(0,ddof=1).sum()/max(np.square(g.mean(0)).sum(),1e-12)),batches=len(g))


def run():
    torch.set_num_threads(CFG['threads'])
    out=HERE/'results';out.mkdir(exist_ok=True)
    frame,x,y=arrays()
    tr=np.flatnonzero((frame.split=='train')&(frame.cohort=='main')); va=np.flatnonzero((frame.split=='validation')&(frame.cohort=='main'))
    te=np.flatnonzero((frame.split=='test')&(frame.cohort=='main')); transfer=np.flatnonzero(frame.cohort=='transfer')
    selected=json.loads((HERE.parent/'lab02_multimodal/results/selection.json').read_text(encoding='utf-8'))['scheme']
    # Train all-branch schema to expose all parameter groups; transfer to lab02 winner (or numbers if already all).
    scheme='all'
    lock(out/'protocol.json',dict(config=CFG,scheme=scheme,transfer_scheme=selected if selected!='all' else 'numbers',split_hash=hashlib.sha256(frame[['id','split','cohort']].to_csv(index=False).encode()).hexdigest()))
    trials=[];curves=[];runs=[];noise=[];models={}
    for opt in CFG['optimizers']:
        for lr in CFG['learning_rates']:
            model,curve,r=fit(scheme,11,x,y,tr,va,steps=CFG['steps'],optimizer=opt,lr=lr)
            trials.append(dict(optimizer=opt,learning_rate=lr,**r))
            print('TUNE',opt,lr,r['status'],r.get('val_macro_f1'),flush=True)
        eligible=successful(pd.DataFrame(trials)).query('optimizer == @opt')
        lr=float(eligible.sort_values('val_macro_f1',ascending=False).iloc[0].learning_rate)
        for seed in CFG['seeds']:
            model,curve,r=fit(scheme,seed,x,y,tr,va,steps=CFG['steps'],optimizer=opt,lr=lr)
            reached=[c['elapsed'] for c in curve if c['val_macro_f1']>=CFG['target_validation_f1']]
            r.update(optimizer=opt,learning_rate=lr,time_to_target_s=min(reached) if reached else None,split_hash=hashlib.sha256(frame[['id','split']].to_csv(index=False).encode()).hexdigest())
            runs.append(r); curves.extend([dict(optimizer=opt,seed=seed,**c) for c in curve])
            noise.append(dict(optimizer=opt,seed=seed,**gradient_noise(model,x,y,tr)))
            models[(opt,seed)]=model
            torch.save(model.state_dict(),out/f'{opt}_{seed}.pt')
    pd.DataFrame(trials).to_csv(out/'sensitivity.csv',index=False)
    pd.DataFrame(runs).to_csv(out/'runs.csv',index=False)
    pd.DataFrame(curves).to_csv(out/'learning.csv',index=False)
    pd.DataFrame(noise).to_csv(out/'gradient_noise.csv',index=False)
    complete=successful(pd.DataFrame(runs))
    v=complete.groupby('optimizer').agg(mean_f1=('val_macro_f1','mean'),sd_f1=('val_macro_f1','std'),elapsed=('elapsed','mean'),n=('seed','count')).reset_index()
    v=v[v.n>=4]
    winner=v[v.mean_f1>=v.mean_f1.max()-.01].sort_values(['sd_f1','elapsed']).iloc[0].optimizer
    save(out/'selection.json',dict(optimizer=winner,selected_on='validation',rule=CFG['selection']))
    metrics=[];predictions=[]
    for (opt,seed),model in models.items():
        for cohort,idx in [('main',te),('transfer',transfer)]:
            p=infer(model,x,idx)
            metrics.append(dict(optimizer=opt,seed=seed,cohort=cohort,**classification(y[idx].numpy(),p)))
            predictions.append(frame.iloc[idx][['id','household','week','label']].assign(optimizer=opt,seed=seed,cohort=cohort,p=p[:,1]))
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    pd.concat(predictions).to_csv(out/'predictions.csv',index=False)
    summary=[]
    for opt,g in pd.DataFrame(metrics).query("cohort == 'main'").groupby('optimizer'):
        low,high=interval(g.macro_f1.to_numpy(),11)
        summary.append(dict(optimizer=opt,mean_f1=float(g.macro_f1.mean()),sd_f1=float(g.macro_f1.std()),ci_low=low,ci_high=high,n_seeds=len(g)))
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    target=selected if selected!='all' else 'numbers'
    transferred=[]
    for opt in dict.fromkeys([winner,'AdamW' if winner!='AdamW' else 'SGD']):
        lr=float(pd.DataFrame(runs).query('optimizer == @opt').iloc[0].learning_rate)
        for seed in CFG['seeds']:
            model,_,r=fit(target,seed,x,y,tr,va,optimizer=opt,lr=lr)
            p=infer(model,x,te)
            transferred.append(dict(optimizer=opt,scheme=target,seed=seed,**classification(y[te].numpy(),p)))
    pd.DataFrame(transferred).to_csv(out/'transfer.csv',index=False)
    # Block bootstrap resamples contiguous weeks; not iid customers or individual transactions.
    pred=pd.concat(predictions).query('optimizer == @winner and seed == 11 and cohort == "main"')
    weeks=sorted(pred.week.unique()); blocks=[weeks[i:i+2] for i in range(0,len(weeks),2)]
    rng=np.random.default_rng(11); bs=[]
    for _ in range(1000):
        g=pd.concat([pred[pred.week.isin(blocks[i])] for i in rng.integers(len(blocks),size=len(blocks))])
        bs.append(classification(g.label.to_numpy(),np.column_stack([1-g.p,g.p]))['macro_f1'])
    save(out/'block_bootstrap.json',dict(block_weeks=2,blocks=len(blocks),ci=np.quantile(bs,[.025,.975]).tolist(),warning='Only six test weeks; few temporal blocks, uncertain interval'))
    model=models[(winner,11)]
    scenarios=[]
    for seed in CFG['seeds']:
        rng=np.random.default_rng(seed); state=0; states=[]
        for week in weeks:
            states.append(state); state=int(rng.choice(2,p=CFG['markov_transition'][state]))
        for name in ['normal','low_demand','markov_switching']:
            altered=[t.clone() for t in x]
            for week,state in zip(weeks,states):
                idx=np.flatnonzero((frame.week==week)&(frame.split=='test')&(frame.cohort=='main'))
                factor=.5 if name=='low_demand' or (name=='markov_switching' and state==1) else 1.
                # Numeric covariate stress, labels unchanged: robustness, not a claim about counterfactual purchases.
                altered[1][idx,:8]*=factor
            p=infer(model,altered,te)
            scenarios.append(dict(seed=seed,scenario=name,states=json.dumps(states),interpretation='numeric covariate shift with fixed observed labels',**classification(y[te].numpy(),p)))
    pd.DataFrame(scenarios).to_csv(out/'scenarios.csv',index=False)
    print('SELECTED',winner,flush=True)
    resources()


if __name__=='__main__':resources() if '--resources' in sys.argv else run()
