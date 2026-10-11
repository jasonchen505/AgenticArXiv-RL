"""Resume only the 23 unstarted registered R034 inference jobs, no new learning."""
import os,sys,fcntl,signal,subprocess,datetime
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';sys.path[:0]=['AgenticArxiv','scripts']
from round034_core import R,D,PY,read,write,verify,sha
import run_round027_retention as h
S=R.parent/'round-035-2026-10-10'
def state(phase,**kw):write(S/'operation-status.json',dict(phase=phase,updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw))
def main():
 h.state=state;signal.signal(signal.SIGTERM,h.terminate);signal.signal(signal.SIGINT,h.terminate)
 with (R/'experiment.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p=read(R/'experiment-plan.json');amendment=read(S/'operation-amendment.json');launch=read(S/'operation-launch.json');verify(p);verify(amendment);assert sha(S/'CPU-recovery-receipt.json')==launch['CPU_recovery_receipt_SHA256'];verify(read(S/'CPU-recovery-receipt.json'))
  for label,s in p['eval_jobs'].items():
   jp=R/(label+'-job.json')
   if label=='source-plan-TRAIN-s49':assert read(jp)['exit_code']==0;continue
   if label=='mixed49-plan-TRAIN-s49':assert read(jp)['exit_code']==1;continue
   assert not jp.exists();assert label in amendment['unstarted_original_labels'];gpu=h.wait_gpu(False,label);verify(p);verify(amendment)
   for f,v in read(R/'all-finals-before-inference.json')['sha256'].items():assert sha(f)==v
   root=D/label;env=dict(os.environ,TMPDIR=str((D/'tmp').resolve()),FIGURE_ANALYSIS_BACKEND='extractive')
   for k,n in [('PDF_RAW_PATH','raw'),('PDF_TRANSLATED_PATH','translated'),('PDF_FIGURES_PATH','figures')]:env[k]=str((root/n).resolve())
   state('evaluating',current_job=label,physical_gpu=gpu);h.run_job([PY,'scripts/run_round020_gpu_job.py','--record',str(jp),'--log',str(R/(label+'.log')),'--gpu',str(gpu),'--',PY,'-u','scripts/round035_eval_entry.py','--label',label],3600,env)
  state('CPU_finalization');subprocess.run([PY,'scripts/finalize_round035_operational.py'],timeout=1800,check=True);state('completed')
if __name__=='__main__':
 try:main()
 except Exception as e:state('failed',error=str(e));raise
