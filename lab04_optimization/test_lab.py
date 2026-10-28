"""Catches different initial states/order across optimizers and incomplete-run contamination."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from lab02_multimodal.run import fit
from lab04_optimization.run import successful

torch.set_num_threads(2)
torch.manual_seed(1)
x=(torch.randn(16,8),torch.randn(16,12),torch.randn(16,3,16,16)); y=torch.tensor([0,1]*8)
hashes=[]
for name in ['SGD','Momentum','Nesterov','Adagrad','RMSprop','Adam','AdamW','Adadelta']:
    _,_,r=fit('all',11,x,y,np.arange(12),np.arange(12,16),steps=2,optimizer=name)
    hashes.append((r['initial_state_hash'],r['batch_order_hash']))
assert len(set(hashes))==1
_,_,r=fit('all',22,x,y,np.arange(12),np.arange(12,16),steps=2)
assert r['initial_state_hash']!=hashes[0][0]
assert len(successful(pd.DataFrame([{'status':'complete','score':1},{'status':'interrupted','score':100}])))==1
memory=Path(__file__).resolve().parent/'results/memory.csv'
if memory.exists():
    m=pd.read_csv(memory).set_index('optimizer')
    assert len(m)==8 and m.loc['AdamW','optimizer_state_bytes']>m.loc['SGD','optimizer_state_bytes']
    assert (m.total_tensor_bytes==m.parameter_bytes+m.gradient_bytes+m.optimizer_state_bytes).all()
print('PASS lab04: eight optimizers, equal initialization/batches, distinct seed, incomplete-run exclusion, tensor storage')
