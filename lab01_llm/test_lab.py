"""Catches silent acceptance of malformed LLM output and dropped invalid rows."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import numpy as np
from lab01_llm.run import parse_prediction, summarize, binary_probability
from lab01_llm.run import HERE,validate_records
from lab01_llm.run import quantize
import json
import torch,weakref,gc
from unittest.mock import patch

assert parse_prediction('{"p":0.75}') == .75
for raw in ['{"p":', '{"p":NaN}', '{"p":1.2}', '{"p":true}', '{"probability":0.5}', '{"p":0.5,"label":1}']:
    try:
        parse_prediction(raw)
        raise AssertionError(raw)
    except ValueError:
        pass
d = pd.DataFrame([dict(label=1, probability=.9, valid=True, elapsed=1), dict(label=1, probability=None, valid=False, elapsed=1)])
r = summarize(d)
assert r['n'] == 2 and r['invalid_rate'] == .5 and r['accuracy'] == .5
assert abs(binary_probability(np.array([[0.,2.,0.]]),0,1)[0]-.8807970779778823)<1e-12
model=torch.nn.Sequential(torch.nn.Sequential(torch.nn.Linear(8,8)),torch.nn.ReLU(),torch.nn.Sequential(torch.nn.Linear(8,2))).eval()
refs=[weakref.ref(m) for m in model.modules() if isinstance(m,torch.nn.Linear)]
original=torch.ao.nn.quantized.dynamic.Linear.from_float
calls=[]
def checked(layer):
    if calls:assert refs[0]() is None,'completed parent retains old float layer'
    calls.append(1)
    return original(layer)
with patch.object(torch.ao.nn.quantized.dynamic.Linear,'from_float',new=checked):quantize(model)
gc.collect()
assert all(ref() is None for ref in refs),'old float layers retained'
assert torch.isfinite(model(torch.ones(2,8))).all()
path=HERE/'results/predictions.jsonl'
if path.exists():
    row=json.loads(path.read_text(encoding='utf-8').splitlines()[0])
    for records in [[row,row],[dict(row,status='failed')],[dict(row,protocol_hash='stale')],[dict(row,label=1-row['label'])]]:
        try:
            validate_records(records)
            raise AssertionError('duplicate/failed/stale/mismatched prediction accepted')
        except ValueError:pass
print('PASS lab01: strict JSON, NaN/range/type rejection, invalid row penalized')
