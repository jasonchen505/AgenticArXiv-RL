#!/usr/bin/env python3
"""CPU official metadata for second training world; fixed2024 window before outputs."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,re,subprocess,time
R=Path('artifacts/reproduction/round-021-2026-10-06');W=R.parent/'round-017-2026-10-06'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
write=lambda p,x:Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
base=lambda p:re.sub(r'v\d+$','',p['id'])
def main():
    plan=read(R/'trainB-metadata-amendment.json')
    for p,h in plan['sha256'].items():assert sha(p)==h,p
    dest=R/'trainB-metadata-candidates';dest.mkdir(exist_ok=True);pools=[]
    with (R/'trainB-metadata.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for i,spec in enumerate(plan['queries']):
            out=dest/f'pool-{i:02d}.json'
            write(R/'train-world-metadata-status.json',{'phase':'trainB_metadata_query','completed_new_train_worlds':1,'trainA_unique_papers':read(R/'trainA-paper-bank.json')['unique_base_ids'],'completed_trainB_pools':len(pools),'current_pool':spec['label'],'GPU_jobs':0,'new_training_steps':0,'gold_passed':False})
            if not out.exists():subprocess.run(['.venv-repro/bin/python','scripts/prepare_round017_paper_bank.py','--query',spec['query'],'--output',str(out)],check=True,timeout=90,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
            raw=read(out);assert raw['query']==spec['query'];selected=[];seen=set()
            for p in raw['results']:
                identifier=base(p)
                if identifier in plan['exclude_base_ids'] or identifier in seen:continue
                assert p['published'][:4]=='2024'
                selected.append(p);seen.add(identifier)
                if len(selected)==10:break
            assert len(selected)==10,(spec['label'],len(selected))
            pools.append({**spec,'papers':selected,'source_sha256':sha(out)})
        ids={base(p) for pool in pools for p in pool['papers']};assert not ids&set(plan['exclude_base_ids'])
        write(R/'trainB-paper-bank.json',{'world':'trainB','pools':pools,'unique_base_ids':len(ids),'overlap_with_original_or_prior_diagnostic_world':0,'overlap_with_trainA':0,'selected_by_model_scores':False,'PDF_resource_gate_passed':False,'gold_gate_passed':False})
        write(R/'train-world-metadata-status.json',{'phase':'metadata_admitted','training_worlds':2,'pools_each':10,'papers_per_pool':10,'unique_papers_each':[read(R/'trainA-paper-bank.json')['unique_base_ids'],len(ids)],'prior_world_overlap':0,'between_new_world_overlap':0,'source_windows':['2025','2024'],'GPU_jobs':0,'new_training_steps':0,'PDFs_acquired':False,'gold_passed':False})
        # Main scientific direction is registered separately before any model outputs.
if __name__=='__main__':
    try:main()
    except BlockingIOError:raise SystemExit('Metadata worker already owns lock')
    except Exception as exc:write(R/'train-world-metadata-status.json',{'phase':'failed','error':str(exc),'GPU_jobs':0,'new_training_steps':0,'gold_passed':False});raise

