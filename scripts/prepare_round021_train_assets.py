#!/usr/bin/env python3
"""Finite CPU real assets for train-only worlds; gold/learning remain separate gates."""
from pathlib import Path
import argparse,datetime,fcntl,hashlib,json,os,subprocess,sys,time
R=Path('artifacts/reproduction/round-021-2026-10-06')
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
write=lambda p,x:Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def state(phase,**kw):write(R/'train-assets-status.json',{'phase':phase,'updated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'GPU_jobs':0,'new_training_steps':0,'gold_passed':False,**kw})
def child(world):
    D=Path('data/repro_round021')/world;D.mkdir(parents=True,exist_ok=True)
    assert not (D/'snapshot.json').exists()
    bank=read(R/(world+'-paper-bank.json'));assert bank['overlap_with_original_or_prior_diagnostic_world']==0
    os.environ.update(STORE_BACKEND='memory',FIGURE_ANALYSIS_BACKEND='extractive',CUDA_VISIBLE_DEVICES='')
    for key,name in [('PDF_RAW_PATH','raw'),('PDF_TRANSLATED_PATH','translated'),('PDF_FIGURES_PATH','figures')]:os.environ[key]=str((D/name).resolve())
    sys.path.insert(0,'AgenticArxiv')
    from tools.bootstrap import require_all_tools
    require_all_tools('Train-only paper-world CPU assets')
    from rl.env import MockArxivEnv
    import rl.build_snapshot as builder
    from rl.backfill_figure_analysis import backfill_entries
    env=MockArxivEnv(snapshot_path=D/'snapshot.json',mode='record',offline_download=False)
    for pool in bank['pools']:
        args=pool['args'];key=env._keyword_search_key(args) if pool['tool']=='search_arxiv_papers' else env._make_key(args)
        env.snapshot.setdefault(pool['tool'],{})[key]={'args':args,'result':pool['papers']}
    papers=builder._unique_snapshot_papers(env);assert len(papers)==bank['unique_base_ids']
    fetched,cached,failed=builder._prefetch_pdfs(papers,4,budget_seconds=300)
    assert failed==0,('Real PDF acquisition failed; retainpartialassets/no fabricated world',failed)
    # Existing T2/T3/T4 extraction remains identical; avoid repeated raster extraction
    # for T5 by using the already tested pure caption backfill for its full enum surface.
    builder.FIGURE_QUESTIONS=()
    ok,fail=builder._snapshot_paper_content(env);assert fail==0 and ok==len(papers)
    snapshot,stats=backfill_entries(env.snapshot,backend='extractive');assert stats.failed==0
    env.snapshot=snapshot;env.save_snapshot()
    write(R/(world+'-asset-manifest.json'),{'phase':'assets_ready_not_gold','world':world,'metadata_unique_papers':bank['unique_base_ids'],'actual_PDFs':len(list((D/'raw').glob('*.pdf'))),'content_papers':ok,'figures':stats.figures,'T5_entries':stats.added,'PDF_downloaded':fetched,'PDF_cached':cached,'gold_passed':False,'new_training_steps':0,'sha256':{str(p):sha(p) for p in D.rglob('*') if p.is_file()}})
def main():
    with (R/'train-assets.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        deadline=time.monotonic()+900
        while not (R/'trainB-paper-bank.json').exists():
            status=read(R/'train-world-metadata-status.json') if (R/'train-world-metadata-status.json').exists() else {}
            assert status.get('phase')!='failed',status
            state('waiting_for_trainB_metadata',metadata_phase=status.get('phase'))
            assert time.monotonic()<deadline,'Finite metadata wait expired'
            time.sleep(15)
        paths=[R/'trainA-paper-bank.json',R/'trainB-paper-bank.json',R/'next-mainline-plan.json',Path(__file__),Path('AgenticArxiv/rl/build_snapshot.py'),Path('AgenticArxiv/rl/backfill_figure_analysis.py'),Path('AgenticArxiv/tools/figure_analysis_tool.py')]
        write(R/'train-asset-plan.json',{'registered_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_only':True,'new_GPU_jobs':0,'all_selected_train_metadata_resources':True,'T5':'Same extractive outputs via complete pure T4-caption backfill; no repeated figure extraction per question','PDF_and_content_gate':'Every metadata-selected paper must have realPDF/content; otherwise fail,retainpartialassets. No fake substitutes.','gold_and_training_not_claimed':True,'sha256':{str(p):sha(p) for p in paths}})
        for world in ['trainA','trainB']:
            target=R/(world+'-asset-manifest.json')
            if target.exists():continue
            state('acquiring_train_assets',world=world)
            with (R/(world+'-assets.log')).open('w') as log:subprocess.run(['.venv-repro/bin/python','-u',__file__,'--world',world],check=True,timeout=1200,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=log)
        state('assets_ready_not_gold',worlds=2,actual_PDFs=[read(R/(w+'-asset-manifest.json'))['actual_PDFs'] for w in ['trainA','trainB']])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--world',choices=['trainA','trainB']);a=p.parse_args()
    if a.world:child(a.world)
    else:
        try:main()
        except BlockingIOError:raise SystemExit('Asset worker already owns lock')
        except Exception as exc:state('failed',error=str(exc));raise

