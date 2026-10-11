import json,hashlib,random
from pathlib import Path
R=Path('artifacts/reproduction/round-034-2026-10-10');D=Path('data/repro_round034');PY='.venv-repro/bin/python'
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def verify(p):
 for f,h in p['sha256'].items():assert sha(f)==h,f
def encode(tok,row):
 prompt=tok.apply_chat_template(row['prompt'],add_generation_prompt=True,tokenize=True);full=tok.apply_chat_template(row['prompt']+row['chosen'],tokenize=True);assert full[:len(prompt)]==prompt and 0<len(prompt)<len(full)<4096
 return {'input_ids':full,'attention_mask':[1]*len(full),'labels':[-100]*len(prompt)+full[len(prompt):]}
