#!/usr/bin/env python3
from pathlib import Path
import datetime,json,re,hashlib
R=Path('artifacts/reproduction/round-024-2026-10-07');read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 status=read(R/'experiment-status.json');base=read(R/'difficulty-status.json');complete=list(Path('outputs/repro_round024').glob('*/final/training_manifest.json'));steps=sum(read(p)['actual_steps'] for p in complete);logged=0
 if status['phase']=='training':
  p=R/(status['arm']+'-train.log')
  if p.exists():logged=max([int(x) for x in re.findall(r'(?<!\d)(\d+)/199',p.read_text(errors='replace'))]+[0]);assert logged<=199
 outcome=read(R/'source-difficulty-outcome.json') if (R/'source-difficulty-outcome.json').exists() else None
 job=read(R/'source-new-dev-job.json') if (R/'source-new-dev-job.json').exists() else None
 progress=None
 if not outcome and (R/'source-new-dev.log').exists():progress=max([int(x) for x in re.findall(r'\[(\d+)/48\]',(R/'source-new-dev.log').read_text(errors='replace'))]+[0])
 plan=read(R/'experiment-plan.json');assert all(sha(p)==h for p,h in plan['sha256'].items());old=read(R/'closure-independent-audit.json');assert old['independent_trials_rescored']==780 and old['full_outcome_verified']
 snapshot={'captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'world_recipe_closed':True,'old_experiment_independent_trials':780,'old_experiment_training_steps':1240,'new_experiment_status':status,'difficulty_status':base,'source_difficulty_outcome':outcome,'source_job_at_capture':job,'source_logged_trial_started':progress,'new_completed_training_finals':len(complete),'new_completed_training_steps':steps,'new_logged_current_arm_training_steps':logged,'new_observed_training_steps':steps+logged,'new_candidate_gains_available':False,'frozen_new_inputs_unchanged':True,'old_outcome_sha256':sha(R/'world-data-outcome.json'),'new_plan_sha256':sha(R/'experiment-plan.json')}
 (R/'experiment-at-report.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in snapshot.items() if k!='source_job_at_capture'},ensure_ascii=False))
if __name__=='__main__':main()
