import os,sys,datetime,fcntl,signal,subprocess
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,'scripts')
from round031_study_core import read,write,PY
from run_round032_parity import N,verify
import run_round027_retention as h

def state(phase,**kw):write(N/'parity-status.json',dict(phase=phase,updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw))
def main():
 h.state=state;signal.signal(signal.SIGTERM,h.terminate);signal.signal(signal.SIGINT,h.terminate)
 with (N/'parity.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p=read(N/'parity-plan.json');verify(p);assert not (N/'parity-outcome.json').exists()
  for mode in p['modes']:
   jp=N/('parity-'+mode+'-job.json');assert not jp.exists();gpu=h.wait_gpu(False,mode);verify(p);root=Path('data/repro_round032')/mode;env=dict(os.environ,TMPDIR=str(Path('data/repro_round031/tmp').resolve()),FIGURE_ANALYSIS_BACKEND='extractive')
   for k,n in [('PDF_RAW_PATH','raw'),('PDF_TRANSLATED_PATH','translated'),('PDF_FIGURES_PATH','figures')]:env[k]=str((root/n).resolve())
   state('evaluating_TRAIN_infra_sanity',current_job=mode,physical_gpu=gpu);h.run_job([PY,'scripts/run_round020_gpu_job.py','--record',str(jp),'--log',str(N/('parity-'+mode+'.log')),'--gpu',str(gpu),'--',PY,'-u','scripts/run_round032_parity.py','--mode',mode],5400,env)
  state('CPU_finalization');subprocess.run([PY,'scripts/finalize_round032_parity.py'],timeout=600,check=True);state('completed')
if __name__=='__main__':
 try:main()
 except Exception as e:state('failed',error=str(e));raise
