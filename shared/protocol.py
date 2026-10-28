"""Small durable journals and stale-result guards."""
import hashlib
import json
from contextlib import contextmanager
from shared.metrics import save


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(1024*1024):h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def lock(path,protocol):
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=protocol:
        raise ValueError(f'Stale protocol: {path}; use a separate result directory')
    save(path,protocol)
    return fingerprint(protocol)


@contextmanager
def journal(path,protocol):
    record=dict(status='running',fingerprint=fingerprint(protocol),protocol=protocol)
    save(path,record)
    try:
        yield record
    except BaseException as e:
        record.update(status='failed',error=type(e).__name__)
        save(path,record)
        raise
    else:
        record['status']=record.get('result',{}).get('status','complete')
        save(path,record)


def analysis_inputs(folder,source):
    return {name:digest(folder/name) for name in ('selection.json',source)}


def check_analysis(root):
    for lab,result,source in [('lab01_llm','paired_baseline.json','predictions.jsonl'),('lab03_marl','paired_uniform.json','episodes.csv'),('lab04_optimization','paired_transfer.json','transfer.csv')]:
        folder=root/lab/'results'
        if json.loads((folder/result).read_text(encoding='utf-8')).get('inputs')!=analysis_inputs(folder,source):
            raise ValueError('Stale analysis; run shared/analysis.py before building reports')
