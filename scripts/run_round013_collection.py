#!/usr/bin/env python3
"""Finite GPU4-only source-SFT collection; weights must not change."""
from pathlib import Path
from datetime import datetime,timezone
import copy,fcntl,hashlib,json,os,subprocess,sys,time
R=Path('artifacts/reproduction/round-013-2026-10-05');PY='.venv-repro/bin/python'
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def state(phase,**x):
    write(R/'collection-status.json',{'phase':phase,'updated_at_utc':datetime.now(timezone.utc).isoformat(),**x})
    if Path('scripts/build_round013_report.py').exists():subprocess.run([PY,'scripts/build_round013_report.py'],check=True)
def worker():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='4';plan=read(R/'collection-plan.json')
    for p,h in plan['sha256'].items():assert sha(p)==h,p
    assert plan['learning_rate']==0. and not Path('outputs/repro_round013/collection').exists()
    tasks=read(R/'collection-tasks.json')
    sys.path.insert(0,'AgenticArxiv')
    from rl import train_grpo as module
    def load(task_set,split):
        assert task_set=='expanded' and split==str(R/'collection-tasks.json')+':train_only'
        return copy.deepcopy(tasks)
    module._load_tasks=load
    module.main(model=plan['source_model'],output_dir='outputs/repro_round013/collection',epochs=2,max_steps=274,batch_size=4,grad_accum=1,lr=0.,beta=.04,reward_curriculum_steps=0,reward_objective='legacy',num_generations=4,max_completion_length=2048,max_turns=6,temperature=1.,loss_type='grpo',mask_truncated_completions=True,importance_sampling_level='token',scale_rewards='group',dynamic_sampling=False,seed=46,snapshot='data/repro_20261002/snapshot_curated_v1.json',task_set='expanded',split=str(R/'collection-tasks.json')+':train_only',allow_zero_variance=True,canary_steps=0,verify=False,report_to='none',save_rollout_traces=True)
def main():
    with (R/'controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=read(R/'collection-plan.json')
        for p,h in plan['sha256'].items():assert sha(p)==h,p
        record=R/'collection-job.json'
        if not record.exists():
            deadline=time.monotonic()+1800
            while subprocess.check_output(['nvidia-smi','--id=4','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                state('waiting_for_gpu4')
                if time.monotonic()>deadline:raise RuntimeError('GPU4 unavailable for30minutes; preserve plan without using other GPUs')
                time.sleep(30)
            scratch=Path('data/repro_round013/collection_scratch');scratch.mkdir(parents=True,exist_ok=True);env=os.environ.copy()
            for key,name in [('PDF_RAW_PATH','pdf_raw'),('PDF_TRANSLATED_PATH','pdf_translated'),('PDF_FIGURES_PATH','pdf_figures')]:env[key]=str((scratch/name).resolve())
            state('running',activity='Frozen source-SFT sampling; lr0, no learned weights')
            subprocess.run([PY,'scripts/run_repro_job.py','--record',str(record),'--log',str(R/'collection.log'),'--gpu','4','--',PY,'-u','scripts/run_round013_collection.py','worker'],env=env,check=True)
        assert read(record).get('exit_code')==0
        state('finalizing');subprocess.run([PY,'scripts/finalize_round013_collection.py'],check=True);state('completed')
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='worker':worker()
    else:
        try:main()
        except BlockingIOError:raise SystemExit('Existing Round013 collection controller holds lock; no duplicate started')
        except Exception as exc:state('failed_or_wait_timeout',error=str(exc));raise
