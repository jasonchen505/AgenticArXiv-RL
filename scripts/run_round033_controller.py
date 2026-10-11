import os,sys,datetime,fcntl,signal,subprocess
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,'scripts')
from round031_study_core import read,write,PY
from run_round033_pilot import R,verify
import run_round027_retention as h
def state(phase,**kw):write(R/'pilot-status.json',dict(phase=phase,updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw))
def main():
 h.state=state;signal.signal(signal.SIGTERM,h.terminate);signal.signal(signal.SIGINT,h.terminate)
 with (R/'pilot.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p=read(R/'pilot-plan.json');verify(p);assert not (R/'pilot-outcome.json').exists()
  for label in p['job_order']:
   jp=R/(label+'-job.json');assert not jp.exists();gpu=h.wait_gpu(False,label);verify(p);root=Path('data/repro_round033')/label;env=dict(os.environ,TMPDIR=str(Path('data/repro_round033/tmp').resolve()),FIGURE_ANALYSIS_BACKEND='extractive')
   for k,n in [('PDF_RAW_PATH','raw'),('PDF_TRANSLATED_PATH','translated'),('PDF_FIGURES_PATH','figures')]:env[k]=str((root/n).resolve())
   state('evaluating_TRAIN_feasibility',current_job=label,physical_gpu=gpu);h.run_job([PY,'scripts/run_round020_gpu_job.py','--record',str(jp),'--log',str(R/(label+'.log')),'--gpu',str(gpu),'--',PY,'-u','scripts/run_round033_pilot.py','--job',label],5400,env)
  state('CPU_finalization');subprocess.run([PY,'scripts/finalize_round033_pilot.py'],timeout=900,check=True);state('completed')
if __name__=='__main__':
 try:main()
 except Exception as e:state('failed',error=str(e));raise
