"""Download pinned public model files with Windows curl (system TLS)."""
import json
import re
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lab01_llm.run import CONFIG, HERE
from shared.data import ROOT
from shared.metrics import save

names=sys.argv[1:] or CONFIG['models']
if any(name not in CONFIG['models'] for name in names):raise ValueError('model must be in fixed experiment config')
manifest_path=HERE/'results'/('model_manifest_'+names[0].replace('/','--')+'.json' if sys.argv[1:] else 'model_manifest.json')
manifest=json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
recorded={}
for saved in (HERE/'results').glob('model_manifest*.json'):
    for name,value in json.loads(saved.read_text(encoding='utf-8')).items():
        if name in recorded and recorded[name]['revision']!=value['revision']:raise ValueError('conflicting model revisions')
        recorded[name]=value
for name in names:
    directory=ROOT/'shared/models'/name.replace('/','--')
    directory.mkdir(parents=True,exist_ok=True)
    info_file=directory/'hub_info.json'
    if not info_file.exists():
        revision_path='/revision/'+recorded[name]['revision'] if name in recorded else ''
        subprocess.run(['curl.exe','--fail','--location','--retry','3','--retry-all-errors','--max-time','300','--silent','--show-error','https://huggingface.co/api/models/'+name+revision_path,'--output',str(info_file)],check=True)
    info=json.loads(info_file.read_text(encoding='utf-8'))
    revision=info['sha']
    if name in recorded and revision!=recorded[name]['revision']:raise ValueError('cached model revision differs from recorded protocol')
    manifest[name]=dict(revision=revision,license=info.get('cardData',{}).get('license'),url='https://huggingface.co/'+name,path=str(directory.relative_to(ROOT)).replace('\\','/'))
    save(manifest_path,manifest)
    for f in info['siblings']:
        n=f['rfilename']
        if not (n.endswith(('.json','.safetensors','.model','.txt','.jinja')) or n=='README.md') or n.startswith(('onnx/','onnx','openvino/')): continue
        path=directory/n
        if path.exists(): continue
        path.parent.mkdir(parents=True,exist_ok=True)
        partial=path.with_suffix(path.suffix+'.partial')
        print('DOWNLOAD',name,n,flush=True)
        if n.endswith('.safetensors'):
            offset=partial.stat().st_size if partial.exists() else 0
            def fetch(start,end):
                chunk=path.with_name(path.name+f'.chunk_{start}'); headers=Path(str(chunk)+'.headers')
                url='https://huggingface.co/'+name+'/resolve/'+revision+'/'+n+'?download=true&offset='+str(start)
                subprocess.run(['curl.exe','--fail','--location','--retry','4','--retry-all-errors','--connect-timeout','15','--max-time','300','--range',f'{start}-{end}','--dump-header',str(headers),'--silent','--show-error',url,'--output',str(chunk)],check=True)
                ranges=re.findall(r'content-range:\s*bytes (\d+)-(\d+)/(\d+)',headers.read_text(encoding='utf-8',errors='replace'),re.I)
                if not ranges: raise ValueError('CDN did not honor Range')
                lo,hi,total=map(int,ranges[-1])
                if lo!=start or chunk.stat().st_size!=hi-lo+1: raise ValueError('range mismatch')
                headers.unlink()
                return chunk,hi+1,total
            # A one-byte range determines total without downloading another complete weight file.
            probe,_,total=fetch(0,0)
            probe.unlink()
            with ThreadPoolExecutor(max_workers=4) as pool:
                jobs=[pool.submit(fetch,start,min(start+16*1024*1024,total)-1) for start in range(offset,total,16*1024*1024)]
                for job in jobs:
                    chunk,stop,total=job.result()
                    with partial.open('ab') as f,chunk.open('rb') as src:
                        while block:=src.read(1024*1024): f.write(block)
                    chunk.unlink()
                    print('CHUNK',name,stop,total,flush=True)
        else:
            subprocess.run(['curl.exe','--fail','--location','--retry','3','--retry-all-errors','--connect-timeout','30','--max-time','120','--silent','--show-error','https://huggingface.co/'+name+'/resolve/'+revision+'/'+n,'--output',str(partial)],check=True)
        if n.endswith('.safetensors'):
            pointer=path.with_suffix('.lfs')
            subprocess.run(['curl.exe','--fail','--location','--retry','4','--retry-all-errors','--max-time','300','--silent','--show-error','https://huggingface.co/'+name+'/raw/'+revision+'/'+n,'--output',str(pointer)],check=True)
            expected=re.search(r'oid sha256:(\w+)',pointer.read_text(encoding='utf-8'))[1]
            digest=hashlib.sha256()
            with partial.open('rb') as src:
                while block:=src.read(8*1024*1024):digest.update(block)
            if digest.hexdigest()!=expected:
                partial.replace(path.with_suffix('.sha256_rejected'))
                raise ValueError('Weight checksum mismatch; rejected file retained, rerun to redownload')
            print('SHA256 VERIFIED',name,n,flush=True)
        partial.replace(path)
    print('READY',name,flush=True)
