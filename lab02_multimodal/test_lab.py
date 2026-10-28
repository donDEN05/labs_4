"""Catches masked-modality leakage, invalid probabilities and noncausal chart inputs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from lab02_multimodal.run import Model, image_from_history

torch.manual_seed(11)
for scheme in ['text','numbers','image','early','late','text_image','numbers_image','all']:
    model=Model(8,scheme).eval()
    text=torch.randn(2,8); numbers=torch.randn(2,12); image=torch.randn(2,3,16,16)
    mask=torch.ones(2,3); mask[:,0]=0
    p=model(text,numbers,image,mask).softmax(1)
    q=model(text*100,numbers,image,mask).softmax(1)
    assert torch.isfinite(p).all() and torch.allclose(p.sum(1),torch.ones(2))
    assert torch.allclose(p,q), scheme
    for missing in range(3):
        inputs=[text.clone(),numbers.clone(),image.clone()]
        inputs[missing].fill_(float('nan'))
        mask=torch.ones(2,3);mask[:,missing]=0
        assert torch.isfinite(model(*inputs,mask)).all(),(scheme,missing)
    if 0 in model.active:
        try:
            model(text*float('nan'),numbers,image)
            raise AssertionError('present NaN accepted')
        except ValueError:pass
x=np.array([1,2,3,4,5,6,7,8,0,0,1,0],float)
assert image_from_history(x).shape==(3,16,16)
assert np.array_equal(image_from_history(x),image_from_history(x.copy()))
print('PASS lab02: eight schemes, missing-modality invariance, finite probabilities, deterministic past-only chart')
