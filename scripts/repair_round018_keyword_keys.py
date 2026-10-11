#!/usr/bin/env python3
from pathlib import Path
import os,sys,json,hashlib,datetime
os.environ.update(STORE_BACKEND='memory',CUDA_VISIBLE_DEVICES='')
sys.path.insert(0,'AgenticArxiv')
from rl.env import MockArxivEnv
R=Path('artifacts/reproduction/round-017-2026-10-06');A=Path('artifacts/reproduction/round-018-2026-10-06');D=Path('data/repro_round017')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
write=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
assert not (R/'cross-world-experiment-plan.json').exists()
s=json.loads((D/'snapshot.json').read_text())
assert sha(D/'snapshot.json')==json.loads((A/'data-repair-audit.json').read_text())['after_snapshot_sha256']
write(A/'keyword-key-repair-plan.json',{'registered_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model_outputs_used':False,'reason':'Original data builder inserted raw query JSON keys instead of existing environment canonical query hash keys. Repair all three registered pools without changing query args, order or paper entries.','before_snapshot_sha256':sha(D/'snapshot.json'),'source_sha256':sha(Path(__file__)),'core_env_sha256':sha(Path('AgenticArxiv/rl/env.py'))})
(A/'snapshot-before-key-repair.json').write_bytes((D/'snapshot.json').read_bytes())
old=s['search_arxiv_papers'];new={MockArxivEnv._keyword_search_key(e['args']):e for e in old.values()}
assert len(new)==len(old)==3
s['search_arxiv_papers']=new
write(D/'snapshot.json',s)
write(A/'keyword-key-repair-audit.json',{'rekeyed_pools':3,'entries_unchanged':True,'other_tool_records_unchanged':True,'model_outputs_used':False,'after_snapshot_sha256':sha(D/'snapshot.json')})

