#!/usr/bin/env python3
from pathlib import Path
import os,datetime,fcntl,subprocess,shutil,signal
os.environ['CUDA_VISIBLE_DEVICES']=''
from round029_online_core import R,PY,read,write,verify,sha,training_audit
from run_round027_retention import wait_gpu,run_job,terminate
# Use original finite GPU admission/job helpers; redirect their status to this study.
import run_round027_retention as helpers

def state(phase,**kw):write(R/'experiment-status.json',{'phase':phase,'updated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw})
def main():
 helpers.state=state;signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
 with (R/'experiment.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p=read(R/'experiment-plan.json');verify(p);assert p['learning_permitted'] and not (R/'online-outcome.json').exists()
  for arm in p['execution_arms']:
   jp=R/(arm+'-train-job.json')
   if jp.exists():assert read(jp).get('exit_code')==0;continue
   gpu=wait_gpu(True,arm);verify(p);state('training' if p['arms'][arm]['lr'] else 'lr0_control',current_job=arm,physical_gpu=gpu);scratch=Path('data/repro_round029')/arm;env=dict(os.environ)
   for k,n in [('PDF_RAW_PATH','raw'),('PDF_TRANSLATED_PATH','translated'),('PDF_FIGURES_PATH','figures')]:env[k]=str((scratch/n).resolve())
   run_job([PY,'scripts/run_round020_gpu_job.py','--record',str(jp),'--log',str(R/(arm+'-train.log')),'--gpu',str(gpu),'--',PY,'-u','scripts/train_round029_online.py',arm],5400,env)
  verify(p);audit=training_audit(p);models={}
  for name,path in p['model_paths'].items():models[name]={'model':path,'sha256':{str(f):sha(f) for f in Path(path).rglob('*') if f.is_file()}}
  freeze=R/'all-finals-before-inference.json'
  if not freeze.exists():write(freeze,{'recorded_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'new_learning_steps':128,'native_optimizer_steps':256,'training_audit':audit,'models':models})
  else:
   for m in read(freeze)['models'].values():
    for f,h in m['sha256'].items():assert sha(f)==h
  for group,g in p['groups'].items():
   for model in p['models_order']:
    if model in p['reused_evals'].get(group,{}):continue
    label=group+'-'+model;jp=R/(label+'-job.json')
    if jp.exists():assert read(jp).get('exit_code')==0;continue
    gpu=wait_gpu(False,label);verify(p)
    for m in models.values():
     for f,h in m['sha256'].items():assert sha(f)==h
    scratch=Path('data/repro_round029/benchmark_scratch')
    if scratch.exists():shutil.rmtree(scratch)
    env=dict(os.environ)
    for k,n in [('PDF_RAW_PATH','raw'),('PDF_TRANSLATED_PATH','translated'),('PDF_FIGURES_PATH','figures')]:env[k]=str((scratch/n).resolve())
    cmd=[PY,'-u','scripts/run_frozen_task_file.py','--tasks',g['tasks'],'--tasks_sha256',p['sha256'][g['tasks']],'--split',g['split'],'--snapshot',g['snapshot'],'--model',p['model_paths'][model],'--output',str(R/label),'--prefix','bench_r20261003_075529','--seed','45','--repeat','3'];state('evaluating',current_job=label,physical_gpu=gpu);run_job([PY,'scripts/run_round020_gpu_job.py','--record',str(jp),'--log',str(R/(label+'.log')),'--gpu',str(gpu),'--']+cmd,3600,env)
  state('CPU_finalization');subprocess.run([PY,'scripts/finalize_round029_online.py'],timeout=1200,check=True)
if __name__=='__main__':
 try:main()
 except Exception as e:state('failed_or_capacity_timeout',error=str(e));raise
