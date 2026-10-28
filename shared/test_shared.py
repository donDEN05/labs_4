"""Run: python shared/test_shared.py. Catches leakage and miscomputed metrics."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from shared.metrics import classification, macro_f1
from shared.data import panel

y = np.array([0, 1, 1, 0])
m = classification(y, np.eye(2)[y])
assert m['macro_f1'] == 1 and m['brier'] == 0
assert abs(macro_f1(np.array([0,0,1,1]),np.array([0,0,1,0]))-11/15)<1e-12
try:
    classification(y, np.array([[1.2, -0.2]] * 4))
    raise AssertionError('invalid probability accepted')
except ValueError:
    pass
rows = pd.DataFrame([dict(household='a', category='c', week=w, quantity=1., sales_value=float(w), retail_disc=0.) for w in range(1, 53)])
p = panel(rows, ['a'], ['c'], window=4)
assert p.iloc[0].week == 5 and p.iloc[0].history_end == 4
for before, after in [('train', 'validation'), ('validation', 'test')]:
    assert p[p.split == before].target_week.max() < p[p.split == after].history_start.min()
changed = rows.copy()
changed.loc[changed.week >= 40, 'sales_value'] = 10000
q = panel(changed, ['a'], ['c'], window=4)
assert np.array_equal(np.stack(p[p.week < 40].numbers), np.stack(q[q.week < 40].numbers))
assert not p[p.week < 40].text.str.contains('10000').any()
print('PASS shared: metrics, invalid probabilities, empty history, temporal split, future invariance')
