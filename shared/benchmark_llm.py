"""Disposable resource probe; results are not part of model comparison."""
import sys,time,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from lab01_llm.run import quantize
torch.set_num_threads(4)
source=Path('shared/models/Qwen--Qwen2.5-0.5B-Instruct')
tok=AutoTokenizer.from_pretrained(source,local_files_only=True)
model=AutoModelForCausalLM.from_pretrained(source,local_files_only=True,dtype=torch.bfloat16).eval()
prompt=tok.apply_chat_template([{'role':'user','content':'Return only JSON {"p":0.5}.'}],tokenize=False,add_generation_prompt=True)
x=tok(prompt,return_tensors='pt')
for mode in ['bfloat16','dynamic_int8']:
    if mode=='dynamic_int8':
        model=quantize(model)
    t=time.perf_counter()
    with torch.inference_mode(): out=model.generate(**x,do_sample=False,max_new_tokens=16,pad_token_id=tok.eos_token_id)
    print(mode,time.perf_counter()-t,tok.decode(out[0,x.input_ids.shape[1]:],skip_special_tokens=True),flush=True)
