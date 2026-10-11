import os,subprocess,datetime
os.environ['CUDA_VISIBLE_DEVICES']=''
from round031_study_core import R,PY,read,write,verify,sha

def main():
 assert not (R/'launch.json').exists() and not list(R.glob('*-job.json'));p=read(R/'experiment-plan.json');verify(p);assert not p['learning_permitted'] and p['total_trial_scores']==2112 and p['inference_jobs']==48 and p['allowed_physical_gpus']==[4,7];assert read(R/'registration-independent-audit.json')['passed'];log=(R/'controller.log').open('a');child=subprocess.Popen(['nohup','timeout','--kill-after=30',str(p['finite_controller_timeout_seconds']),PY,'-u','scripts/run_round031_study.py'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);write(R/'launch.json',{'launched_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':child.pid,'plan_sha256':sha(R/'experiment-plan.json'),'finite_timeout_seconds':86400,'only_physical_gpu':[4,7],'no_periodic_automation':True,'no_learning':True});print(read(R/'launch.json'))
if __name__=='__main__':main()
