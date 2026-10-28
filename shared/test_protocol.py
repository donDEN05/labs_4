"""One failure-path check for the durable training journal."""
import json
import sys
import uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shared.protocol import journal,lock

folder=Path(__file__).resolve().parents[1]/('.protocol_check_'+uuid.uuid4().hex)
folder.mkdir()
try:
    path=folder/'run.json'
    try:
        with journal(path,dict(seed=11)):
            assert json.loads(path.read_text(encoding='utf-8'))['status']=='running'
            raise RuntimeError('simulated interruption')
    except RuntimeError:pass
    assert json.loads(path.read_text(encoding='utf-8'))['status']=='failed'
    lock(path.with_name('protocol.json'),dict(seed=11))
    try:
        lock(path.with_name('protocol.json'),dict(seed=22))
        raise AssertionError('stale protocol accepted')
    except ValueError:pass
finally:
    for item in folder.iterdir():item.unlink()
    folder.rmdir()
print('PASS protocol: durable running/failed status, changed configuration rejected')
