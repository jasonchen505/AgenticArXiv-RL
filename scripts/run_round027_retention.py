#!/usr/bin/env python3
"""Finite, locked four-arm retention study; no early stopping or GPU preemption."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,shutil,signal,subprocess,time
os.environ['CUDA_VISIBLE_DEVICES']=''
R=Path('artifacts/reproduction/round-027-2026-10-08');PY='.venv-repro/bin/python';read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def state(phase,**kw):write(R/'experiment-status.json',dict(phase=phase,updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw))
def verify(plan):
 for p,h in plan['sha256'].items():assert sha(p)==h,p
def admission(training,gpu):
 assert gpu in [4,7];row=[x.strip() for x in subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-gpu=uuid,name,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')];processes=subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True);active=sum(x.split(',')[0].strip()==row[0] for x in processes.splitlines());free=int(row[2]);return row[1]=='NVIDIA GeForce RTX 4090' and free>=(23000 if training else 8192) and (not training or active==0),{'free_mib':free,'UUID_compute_processes':active,'uuid':row[0]}
def wait_gpu(training,label):
 deadline=time.monotonic()+(5400 if training else 1800)
 while True:
  devices={}
  for gpu in ([7,4] if training else [4,7]):
   ok,info=admission(training,gpu);devices[str(gpu)]=info
   if ok:return gpu
  state('waiting_capacity',job=label,training=training,devices=devices);assert time.monotonic()<deadline,'Finite authorized GPU wait expired';time.sleep(30)
def run_job(cmd,timeout,env=None):
 child=subprocess.Popen(cmd,env=env,start_new_session=True)
 try:
  code=child.wait(timeout=timeout);assert code==0,'Owned job exited '+str(code)
 finally:
  if child.poll() is None:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
def terminate(signum,frame):raise RuntimeError('Controller termination signal '+str(signum))
def main():
 signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
 with (R/'experiment.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);plan=read(R/'experiment-plan.json');verify(plan);assert plan['learning_permitted'] and plan['allowed_physical_gpus']==[4,7] and not (R/'retention-outcome.json').exists()
  for name in plan['learning_arms']:
   record=R/(name+'-train-job.json')
   if record.exists():assert read(record).get('exit_code')==0;continue
   gpu=wait_gpu(True,name);verify(plan);state('training',current_job=name,physical_gpu=gpu)
   arm=name[2:-2];seed=name[-2:];cmd=[PY,'-u','scripts/train_round027_retention.py','--arm',arm,'--seed',seed,'--output','outputs/repro_round027/'+name]
   run_job([PY,'scripts/run_round020_gpu_job.py','--record',str(record),'--log',str(R/(name+'-train.log')),'--gpu',str(gpu),'--']+cmd,5400)
  verify(plan)
  import torch
  from safetensors.torch import load_file
  source=load_file(plan['source_model']+'/adapter_model.safetensors');models={}
  for name,path in plan['model_paths'].items():
   d=Path(path)
   if name in plan['learning_arms']:
    m=read(d/'training_manifest.json');a=load_file(str(d/'adapter_model.safetensors'));ref=load_file(str(d/'ref/adapter_model.safetensors'));assert set(a)==set(ref)==set(source) and len(a)==392 and all(torch.equal(source[k],ref[k]) for k in source) and sum(not torch.equal(source[k],a[k]) for k in source)==392
    assert m['dataset_rows']==795 and m['actual_steps']==199 and m['anchor_forward_backward_calls']==795 and m['anchor_exposures_order']==plan['anchor_orders'][str(m['seed'])]
   models[name]={'model':path,'sha256':{str(p):sha(p) for p in d.rglob('*') if p.is_file()}}
  freeze=R/'all-finals-before-inference.json'
  if not freeze.exists():write(freeze,{'recorded_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'new_training_steps':796,'models':models})
  else:
   for model in read(freeze)['models'].values():
    for p,h in model['sha256'].items():assert sha(p)==h,p
  for group,g in plan['groups'].items():
   for model in plan['models_order']:
    if model in plan['reused_evals'].get(group,{}):continue
    label=group+'-'+model;record=R/(label+'-job.json')
    if record.exists():assert read(record).get('exit_code')==0;continue
    gpu=wait_gpu(False,label);verify(plan)
    for m in models.values():
     for p,h in m['sha256'].items():assert sha(p)==h,p
    scratch=Path('data/repro_round027/benchmark_scratch')
    if scratch.exists():shutil.rmtree(scratch)
    env=dict(os.environ)
    for key,name in [('PDF_RAW_PATH','raw'),('PDF_TRANSLATED_PATH','translated'),('PDF_FIGURES_PATH','figures')]:env[key]=str((scratch/name).resolve())
    cmd=[PY,'-u','scripts/run_frozen_task_file.py','--tasks',g['tasks'],'--tasks_sha256',plan['sha256'][g['tasks']],'--split',g['split'],'--snapshot',g['snapshot'],'--model',plan['model_paths'][model],'--output',str(R/label),'--prefix','bench_r20261003_075529','--seed','45','--repeat','3'];state('evaluating',current_job=label,physical_gpu=gpu)
    run_job([PY,'scripts/run_round020_gpu_job.py','--record',str(record),'--log',str(R/(label+'.log')),'--gpu',str(gpu),'--']+cmd,3600,env)
  state('CPU_finalization');verify(plan);subprocess.run([PY,'scripts/finalize_round027_retention.py'],timeout=1200,check=True)
if __name__=='__main__':
 try:main()
 except BlockingIOError:raise SystemExit('Existing retention controller owns lock')
 except Exception as e:state('failed_or_capacity_timeout',error=str(e));raise
