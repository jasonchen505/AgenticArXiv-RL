"""One finite CPU completion process, no inference/retries/periodic automation."""
import os,sys,time,fcntl,subprocess,datetime
os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,'scripts')
from round031_study_core import R,PY,read,write,verify,sha

def main():
 p=read(R/'experiment-plan.json');verify(p);am=read(R/'CPU-completion-amendment.json')
 for f,h in am['sha256'].items():assert sha(f)==h
 deadline=datetime.datetime.fromisoformat(am['original_controller_deadline_utc']).timestamp()
 with (R/'CPU-completion.lock').open('a') as own:
  fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
  while time.time()<deadline:
   if (R/'study-outcome.json').exists():return
   jobs=[R/(g+'-'+m+'-s'+str(s)+'-job.json') for s in p['generation_seeds'] for g in p['groups'] for m in p['modes']]
   if any(f.exists() and read(f).get('exit_code',0)!=0 for f in jobs):raise RuntimeError('Inference failed: no retry or budget change')
   if all(f.exists() and read(f).get('exit_code')==0 for f in jobs):
    with (R/'experiment.lock').open('a') as lock:
     try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
     except BlockingIOError:time.sleep(5);continue
     original=read(R/'experiment-status.json');write(R/'original-status-before-CPU-completion.json',original);verify(p);subprocess.run([PY,'scripts/finalize_round031_bootstrap.py'],timeout=1200,check=True);write(R/'experiment-status.json',{'phase':'completed','updated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'outcome':str(R/'study-outcome.json'),'CPU_operational_bootstrap':str(R/'CPU-completion-amendment.json')});return
   time.sleep(30)
  raise RuntimeError('Original finite experiment deadline reached')
if __name__=='__main__':main()
