#!/usr/bin/env python3
import os,sys,subprocess,datetime
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
from round029_online_core import R,PY,read,write,verify,sha

def main():
 assert not (R/'launch.json').exists() and not any(R.glob('*-train-job.json'));p=read(R/'experiment-plan.json');verify(p);a=read(R/'fresh-independent-audit.json');assert p['learning_permitted'] and a['passed']
 assert sum(p['group_trial_runs'].values())*len(p['models_order'])==p['total_trial_scores']==1320 and 1320-168==1152 and len(p['groups'])*len(p['models_order'])-sum(len(x) for x in p['reused_evals'].values())==26 and 6*2*2==24
 assert p['planned_learning_update_steps']==128 and p['planned_lr0_optimizer_steps']==128 and p['planned_total_native_optimizer_steps']==256 and p['planned_total_rollouts']==1024 and p['allowed_physical_gpus']==[4,7]
 log=(R/'controller.log').open('a');cmd=['nohup','timeout','--kill-after=30',str(p['finite_controller_timeout_seconds']),PY,'-u','scripts/run_round029_online.py'];child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);write(R/'launch.json',{'launched_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':child.pid,'plan_sha256':sha(R/'experiment-plan.json'),'finite_timeout_seconds':p['finite_controller_timeout_seconds'],'budget_independently_enumerated':True,'launcher_source_sha256':sha(Path(__file__)),'only_physical_gpu':[4,7],'no_periodic_automation':True});print(read(R/'launch.json'))
if __name__=='__main__':main()
