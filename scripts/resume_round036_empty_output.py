import os,sys,datetime,subprocess,fcntl,signal,shutil
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';sys.path[:0]=['AgenticArxiv','scripts']
from round036_core import R,D,PY,read,write,sha,verify
import run_round036_pilot as original
import run_round027_retention as h

def state(phase,**kw):write(R/'operation-status.json',dict(phase=phase,updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw))
def resume():
 original.state=state;signal.signal(signal.SIGTERM,h.terminate);signal.signal(signal.SIGINT,h.terminate)
 with (R/'pilot.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p=read(R/'pilot-plan.json');verify(p);verify(read(R/'empty-output-operation-amendment.json'));assert read(R/'preflight-result.json')['optimizer_steps']==0
  # New job record, same frozen trainer and complete recipe. Original exit1 is untouched.
  original.job(p,'plan49-train-recovered',['scripts/train_round036_plan.py','--arm','plan49'],True,5400)
  original.job(p,'lr0_49-train',['scripts/train_round036_plan.py','--arm','lr0_49'],True,5400)
  verify(p);original.freeze(p)
  for label in p['eval_jobs']:original.job(p,label,['scripts/run_round036_eval.py','--label',label],False,3600)
  state('CPU_finalization');subprocess.run([PY,'scripts/finalize_round036_pilot.py'],check=True,timeout=1800);state('completed')
def register():
 assert not (R/'operation-launch.json').exists();p=read(R/'pilot-plan.json');verify(p);failed=read(R/'plan49-train-job.json');assert failed['exit_code']==1;assert read(R/'pilot-status.json')['phase']=='failed';empty=Path(p['model_paths']['plan49']).parent;assert empty.exists() and not list(empty.rglob('*'));assert not Path(p['model_paths']['lr0_49']).parent.exists();preserved=D/'preflight-owned-empty-output';assert not preserved.exists();launch=read(R/'pilot-launch.json');deadline=datetime.datetime.fromisoformat(launch['launched_at_utc'])+datetime.timedelta(seconds=launch['finite_timeout_seconds']);remaining=int((deadline-datetime.datetime.now(datetime.timezone.utc)).total_seconds());assert remaining>0
 failure=R/'original-prelearning-failure';failure.mkdir()
 for n in ['plan49-train-job.json','plan49-train.log','pilot-status.json','pilot-controller.log']:shutil.copyfile(R/n,failure/n)
 write(R/'empty-output-operation-amendment.json',{'registered_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_pilot_plan_SHA256':sha(R/'pilot-plan.json'),'failure':'Trainer construction during zero-step preflight creates an empty candidate output directory; subsequent not-exists guard rejects it before policy load or optimizer work','candidate_optimizer_calls_before_failure':0,'candidate_final_weights_exist':False,'empty_directory_entries':0,'move_owned_empty_directory_from':str(empty),'preserve_at':str(preserved),'original_GPU_exit1_preserved':True,'original_preflight_not_rerun':True,'original_trainer_and_recipe_and_eval_and_rules_unchanged':True,'new_training_job_record':'plan49-train-recovered-job.json','original_deadline_utc':deadline.isoformat(),'sha256':{str(Path('scripts')/'resume_round036_empty_output.py'):sha(Path('scripts')/'resume_round036_empty_output.py'),str(R/'plan49-train-job.json'):sha(R/'plan49-train-job.json'),str(R/'plan49-train.log'):sha(R/'plan49-train.log')}});empty.rename(preserved);log=(R/'operation-controller.log').open('a');child=subprocess.Popen(['nohup','timeout','--kill-after=30',str(remaining),PY,'-u','scripts/resume_round036_empty_output.py','--resume'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);write(R/'operation-launch.json',{'launched_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':child.pid,'remaining_original_deadline_seconds':remaining,'original_deadline_utc':deadline.isoformat(),'amendment_SHA256':sha(R/'empty-output-operation-amendment.json'),'no_periodic_automation':True,'new_scientific_arms':0,'original_failed_job_preserved':True});print(read(R/'operation-launch.json'))
if __name__=='__main__':
 if '--resume' in sys.argv:
  try:resume()
  except Exception as e:state('failed',error=str(e));raise
 else:register()
