from pathlib import Path
import os,datetime,fcntl,subprocess,signal
os.environ['CUDA_VISIBLE_DEVICES']=''
from round031_study_core import R,PY,read,write,verify
import run_round027_retention as helpers

def state(phase,**kw):write(R/'experiment-status.json',{'phase':phase,'updated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw})
def main():
 helpers.state=state;signal.signal(signal.SIGTERM,helpers.terminate);signal.signal(signal.SIGINT,helpers.terminate)
 with (R/'experiment.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p=read(R/'experiment-plan.json');verify(p);assert not p['learning_permitted'] and p['allowed_physical_gpus']==[4,7] and not (R/'study-outcome.json').exists()
  for seed in p['generation_seeds']:
   for group in p['groups']:
    for mode in p['modes']:
     label=group+'-'+mode+'-s'+str(seed);jp=R/(label+'-job.json')
     if jp.exists():assert read(jp).get('exit_code')==0;continue
     gpu=helpers.wait_gpu(False,label);verify(p);scratch=Path('data/repro_round031')/label;env=dict(os.environ,TMPDIR=str(Path('data/repro_round031/tmp').resolve()),FIGURE_ANALYSIS_BACKEND='extractive')
     for key,n in [('PDF_RAW_PATH','raw'),('PDF_TRANSLATED_PATH','translated'),('PDF_FIGURES_PATH','figures')]:env[key]=str((scratch/n).resolve())
     cmd=[PY,'-u','scripts/run_round031_eval.py','--mode',mode,'--group',group,'--seed',str(seed)];state('evaluating',current_job=label,physical_gpu=gpu);helpers.run_job([PY,'scripts/run_round020_gpu_job.py','--record',str(jp),'--log',str(R/(label+'.log')),'--gpu',str(gpu),'--']+cmd,p['inference_job_timeout_seconds'],env)
  state('CPU_finalization');verify(p);subprocess.run([PY,'scripts/finalize_round031_study.py'],timeout=1200,check=True);state('completed',outcome=str(R/'study-outcome.json'))
if __name__=='__main__':
 try:main()
 except Exception as e:state('failed_or_capacity_timeout',error=str(e));raise
