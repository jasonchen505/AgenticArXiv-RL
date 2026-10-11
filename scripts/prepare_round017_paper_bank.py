#!/usr/bin/env python3
"""Finite CPU-only official arXiv metadata discovery; no model outputs."""
from pathlib import Path
import argparse,datetime,fcntl,json,os,re,subprocess,sys,time
R=Path('artifacts/reproduction/round-017-2026-10-06')
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def child(query,out):
    sys.path.insert(0,'AgenticArxiv')
    import arxiv
    from tools.arxiv_tool import _paper_info
    client=arxiv.Client(page_size=50,delay_seconds=3,num_retries=1)
    client.query_url='https://export.arxiv.org/api/query'
    rows=[_paper_info(p) for p in client.results(arxiv.Search(query=query,max_results=50,sort_by=arxiv.SortCriterion.SubmittedDate,sort_order=arxiv.SortOrder.Descending))]
    write(out,{'query':query,'source':'https://export.arxiv.org/api/query','queried_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'results':rows})
def main():
    plan=read(R/'next-stage-plan.json');directory=R/'metadata-candidates';directory.mkdir(exist_ok=True);bank=[]
    from hashlib import sha256
    for p,h in plan['frozen_input_sha256'].items():assert sha256(Path(p).read_bytes()).hexdigest()==h,p
    excluded=set(plan['exclude_prior_world_base_ids']);total_deadline=time.monotonic()+1100
    with (R/'data-preparation.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for i,spec in enumerate(plan['metadata_queries']):
            output=directory/f'pool-{i:02d}.json'
            write(R/'data-preparation-status.json',{'phase':'metadata_discovery','current_pool':spec['label'],'completed_pools':len(bank),'GPU_jobs':0,'model_inference':False})
            if not output.exists():
                remaining=total_deadline-time.monotonic();assert remaining>0
                subprocess.run([sys.executable,__file__,'--query',spec['query'],'--output',str(output)],check=True,timeout=min(90,remaining),env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
                time.sleep(3)
            raw=read(output);assert raw['query']==spec['query']
            chosen=[];seen=set()
            for p in raw['results']:
                base=re.sub(r'v\d+$','',p['id'])
                assert p['published'][:4]=='2025'
                if base in excluded or base in seen:continue
                seen.add(base);chosen.append(p)
                if len(chosen)==10:break
            assert len(chosen)>=spec['minimum_count'],(spec['label'],len(chosen))
            bank.append({**spec,'papers':chosen,'source_result_sha256':sha256(output.read_bytes()).hexdigest()})
        all_ids={re.sub(r'v\d+$','',p['id']) for pool in bank for p in pool['papers']};assert not all_ids&excluded
        write(R/'paper-bank.json',{'pools':bank,'unique_papers':len(all_ids),'overlap_with_prior_world_base_ids':0,'selection_from_model_outputs':False,'PDF_or_tool_resource_validation_complete':False})
        write(R/'data-preparation-status.json',{'phase':'metadata_frozen','completed_pools':len(bank),'unique_papers':len(all_ids),'overlap_with_prior_world_base_ids':0,'GPU_jobs':0,'model_inference':False,'next_gate':'AcquirePDFs/buildnewcaptionworld/CPUgold22of22 beforeanynewmodeloutput.'})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--query');p.add_argument('--output');a=p.parse_args()
    if a.query:child(a.query,a.output)
    else:
        try:main()
        except Exception as exc:write(R/'data-preparation-status.json',{'phase':'failed','error':str(exc),'GPU_jobs':0,'model_inference':False});raise
