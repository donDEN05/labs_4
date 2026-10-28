"""Binary classification, grouped uncertainty and saved experiment records."""
import json
from pathlib import Path
import numpy as np


def macro_f1(y, prediction):
    cm=np.bincount(2*np.asarray(y,int)+np.asarray(prediction,int),minlength=4).reshape(2,2)
    denominator=cm.sum(0)+cm.sum(1)
    return float(np.divide(2*cm.diagonal(),denominator,out=np.zeros(2),where=denominator!=0).mean())


def classification(y, p):
    y, p = np.asarray(y, int), np.asarray(p, float)
    if p.shape != (len(y), 2) or not len(y) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(1), 1):
        raise ValueError('expected finite normalized binary probabilities')
    pred = p.argmax(1)
    cm = np.array([[np.sum((y == a) & (pred == b)) for b in (0, 1)] for a in (0, 1)])
    f1 = [2 * cm[c, c] / max(cm[c].sum() + cm[:, c].sum(), 1) for c in (0, 1)]
    tn, fp, fn, tp = cm.ravel()
    den = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** .5
    conf, correct = p.max(1), pred == y
    ece = sum(abs(correct[(conf >= lo) & (conf < lo + .1 + (lo == .9) * 1e-8)].mean() - conf[(conf >= lo) & (conf < lo + .1 + (lo == .9) * 1e-8)].mean()) * ((conf >= lo) & (conf < lo + .1 + (lo == .9) * 1e-8)).mean() for lo in np.arange(0, 1, .1) if ((conf >= lo) & (conf < lo + .1 + (lo == .9) * 1e-8)).any())
    return dict(macro_f1=float(np.mean(f1)), f1_no=float(f1[0]), f1_buy=float(f1[1]), accuracy=float(correct.mean()), mcc=float((tp * tn - fp * fn) / den) if den else 0., brier=float(np.mean((p[:, 1] - y) ** 2)), ece=float(ece), confusion=cm.tolist(), n=len(y))


def interval(values, seed, repeats=1000):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(repeats, len(values)))].mean(1)
    return tuple(float(x) for x in np.quantile(means, [.025, .975]))


def f1_interval(y, p, groups=None, seed=11, repeats=1000):
    y, p = np.asarray(y), np.asarray(p)
    groups = np.arange(len(y)) if groups is None else np.asarray(groups)
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    vals = []
    blocks = [np.flatnonzero(groups == g) for g in unique]
    for _ in range(repeats):
        idx = np.concatenate([blocks[i] for i in rng.integers(len(blocks), size=len(blocks))])
        vals.append(macro_f1(y[idx],p[idx].argmax(1)))
    return np.quantile(vals, [.025, .975]).tolist()


def paired(y, a, b, groups, seed=11, repeats=1000):
    y, a, b, groups = map(np.asarray, (y, a, b, groups))
    blocks = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    rng, diffs, null = np.random.default_rng(seed), [], []
    observed = macro_f1(y,a.argmax(1)) - macro_f1(y,b.argmax(1))
    for _ in range(repeats):
        idx = np.concatenate([blocks[i] for i in rng.integers(len(blocks), size=len(blocks))])
        diffs.append(macro_f1(y[idx],a[idx].argmax(1))-macro_f1(y[idx],b[idx].argmax(1)))
        swap = np.zeros(len(y), bool)
        for block, flag in zip(blocks, rng.integers(2, size=len(blocks))):
            swap[block] = flag
        aa, bb = np.where(swap[:, None], b, a), np.where(swap[:, None], a, b)
        null.append(macro_f1(y,aa.argmax(1))-macro_f1(y,bb.argmax(1)))
    return dict(delta=float(observed), ci=np.quantile(diffs, [.025, .975]).tolist(), p=float((1 + np.sum(np.abs(null) >= abs(observed))) / (repeats + 1)), groups=len(blocks))


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)
