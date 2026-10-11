#!/usr/bin/env python3
"""Finite CPU-only CL12 capacity admission; retain training program semantics."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,re,subprocess,sys
R=Path('artifacts/reproduction/round-022-2026-10-07');OLD=R.parent/'round-021-2026-10-06';W=R.parent/'round-017-2026-10-06';D=Path('data/repro_round022_resource')
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def base(p):return re.sub(r'v\d+$','',p['id'])
def child(world):
    d=D/world;d.mkdir(parents=True,exist_ok=True)
    os.environ.update(STORE_BACKEND='memory',CUDA_VISIBLE_DEVICES='',FIGURE_ANALYSIS_BACKEND='extractive')
    for key,name in [('PDF_RAW_PATH','raw'),('PDF_TRANSLATED_PATH','translated'),('PDF_FIGURES_PATH','figures')]:os.environ[key]=str((d/name).resolve())
    sys.path[:0]=['AgenticArxiv','scripts']
    from tools.bootstrap import require_all_tools
    require_all_tools('Round022 CL12 actual asset extension')
    from rl.env import MockArxivEnv
    from rl import build_snapshot as builder
    from rl.backfill_figure_analysis import backfill_entries
    builder.FIGURE_QUESTIONS=()
    selected=read(R/(world+'-CL-reserve.json'));env=MockArxivEnv(d/'reserve-snapshot.json',mode='record',offline_download=False)
    args={'aspect':'CL','days':30,'max_results':2};env.snapshot['get_recently_submitted_cs_papers']={env._make_key(args):{'args':args,'result':selected}}
    papers=builder._unique_snapshot_papers(env);got,cached,failed=builder._prefetch_pdfs(papers,2,budget_seconds=240);assert failed==0
    ok,failed=builder._snapshot_paper_content(env);assert ok==2 and failed==0
    updated,stats=backfill_entries(env.snapshot);assert stats['failed']==0;env.snapshot=updated;env.save_snapshot()
    original=read(Path('data/repro_round021')/world/'snapshot.json')
    for key in ['get_paper_content','summarize_paper','extract_paper_figures','analyze_figure']:
        for k,v in env.snapshot.get(key,{}).items():assert k not in original[key];original[key][k]=v
    e=next(e for e in original['get_recently_submitted_cs_papers'].values() if e['args']['aspect']=='CL');assert len(e['result'])==10;e['result']+=selected
    out=D/(world+'-snapshot.json');write(out,original)
    files=[out]+[p for p in d.rglob('*') if p.is_file()]
    write(R/(world+'-reserve-manifest.json'),{'CL_pool_size':12,'actual_additional_PDFs':2,'program_ref12_retained':True,'used_model_outputs':False,'GPU_jobs':0,'new_training_steps':0,'sha256':{str(p):sha(p) for p in files}})
def main():
    D.mkdir(exist_ok=True)
    assert not (R/'CL12-resource-amendment.json').exists()
    banks=[read(OLD/(w+'-paper-bank.json')) for w in ['trainA','trainB']];prior=read(W/'next-stage-plan.json')['exclude_prior_world_base_ids'];held=read(W/'paper-bank.json')
    excluded=set(prior)|{base(p) for b in banks+[held] for pool in b['pools'] for p in pool['papers']};selected={};sources=[]
    for world,folder in [('trainA',W/'metadata-candidates'),('trainB',OLD/'trainB-metadata-candidates')]:
        paths=[p for p in sorted(folder.glob('pool-*.json')) if 'cat:cs.CL' in read(p)['query']];assert len(paths)==1
        source=paths[0];papers=[]
        for p in read(source)['results']:
            if base(p) in excluded:continue
            papers.append(p);excluded.add(base(p))
            if len(papers)==2:break
        assert len(papers)==2;selected[world]=papers;sources.append(source)
    write(R/'CL12-resource-amendment.json',{'registered_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reason':'Gold failed before any model outputs because original train variant requests ordinal12 but admittedCLpool had10. Preserve all137programs and numeric refs; extend onlyCLcapacity to12.','selection_rule':'Eachworld first2eligible distinct candidates from its frozen originalCL50response. Exclude original231/priorraw70/both83+72banks/otherreserve IDs. No model scores, no task removal/ref relaxation.','only_train_program_requirement':'augp1_state_ref_last_of_10_last12','new_base_ids':{w:[base(p) for p in ps] for w,ps in selected.items()},'GPU_jobs':0,'new_training_steps':0,'sha256':{str(p):sha(p) for p in sources+[Path(__file__),R/'train-data-plan.json']}})
    for w,papers in selected.items():write(R/(w+'-CL-reserve.json'),papers)
    for w in ['trainA','trainB']:
        with (R/(w+'-reserve-assets.log')).open('w') as f:subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--world',w],stdout=f,stderr=subprocess.STDOUT,timeout=600,check=True)
    write(R/'CL12-resource-status.json',{'phase':'CL12_assets_ready','extra_PDFs':4,'new_training_steps':0,'GPU_jobs':0})
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--world');args=parser.parse_args()
    if args.world:child(args.world)
    else:main()
