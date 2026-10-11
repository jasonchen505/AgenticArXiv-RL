#!/usr/bin/env python3
from pathlib import Path
import datetime,hashlib,json,os,subprocess
R=Path('artifacts/reproduction/round-027-2026-10-08');read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert not (R/'launch.json').exists() and not Path('outputs/repro_round027').exists();plan=read(R/'experiment-plan.json');pre=read(R/'preflight-result.json');audit=read(R/'integration-independent-audit.json');assert plan['learning_permitted'] and pre['passed'] and audit['passed']
 for p,h in plan['sha256'].items():assert sha(p)==h,p
 assert datetime.datetime.fromisoformat(read(R/'preflight2-job.json')['finished_at_utc'])<datetime.datetime.fromisoformat(plan['registered_at_utc'])
 assert plan['coefficient']==.1 and plan['planned_new_learning_steps']==4*199 and plan['chosen_CE_row_exposures']==plan['anchor_row_exposures']==4*795 and len(plan['model_paths'])==9
 total=new=jobs=0
 for group,g in plan['groups'].items():
  sf,key=g['split'].rsplit(':',1);n=len(read(sf)['split'][key])*3
  for model in plan['models_order']:
   total+=n
   if model not in plan['reused_evals'].get(group,{}):new+=n;jobs+=1
 assert total==1512 and new==1152 and jobs==26 and total-new==360
 command=['nohup','timeout','--signal=TERM','--kill-after=30','43200','.venv-repro/bin/python','-u','scripts/run_round027_retention.py']
 with (R/'controller.log').open('xb') as log:child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
 result={'launched_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':child.pid,'command':command,'finite_timeout_seconds':43200,'plan_sha256':sha(R/'experiment-plan.json'),'training_steps_planned':796,'total_trial_scores':1512,'new_runs':1152,'historical_reused':360,'inference_jobs':26,'comparisons':40,'allowed_physical_gpus':[4,7],'zero_optimizer_step_preflight_passed':True,'budget_independently_enumerated':True,'all_final_checkpoints_before_new_inference':True,'no_automatic_promotion':True}
 (R/'launch.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
