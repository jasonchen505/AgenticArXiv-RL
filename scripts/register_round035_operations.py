import os,sys,json,datetime,subprocess,shutil,ast,hashlib
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,'scripts')
from round034_core import R,read,write,sha,verify
S=R.parent/'round-035-2026-10-10'
def main():
 S.mkdir(exist_ok=True);assert not (S/'operation-amendment.json').exists();p=read(R/'experiment-plan.json');verify(p);freeze=read(R/'all-finals-before-inference.json')
 for f,h in freeze['sha256'].items():assert sha(f)==h
 assert read(R/'experiment-status.json')['phase']=='failed';assert read(R/'source-plan-TRAIN-s49-job.json')['exit_code']==0 and read(R/'mixed49-plan-TRAIN-s49-job.json')['exit_code']==1
 launch=read(R/'launch.json');assert subprocess.run(['kill','-0',str(launch['pid'])],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0
 pending=[k for k in p['eval_jobs'] if not (R/(k+'-job.json')).exists()];assert len(pending)==23
 source=(Path('scripts/finalize_round034_study.py')).read_text();new=(Path('scripts/finalize_round035_operational.py')).read_text();get_rules=lambda text:ast.dump(next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='rules'));assert get_rules(source)==get_rules(new)
 original=S/'original-controller-failure';original.mkdir()
 for name in ['experiment-status.json','controller.log','mixed49-plan-TRAIN-s49-job.json','mixed49-plan-TRAIN-s49.log']:shutil.copyfile(R/name,original/name)
 copies=S/'source-copies';copies.mkdir()
 names=['round035_literal_error_log.py','round035_eval_entry.py','recover_round035_frozen_plans.py','finalize_round035_operational.py','resume_round035_operations.py','register_round035_operations.py','launch_round035_resume.py']
 for name in names:shutil.copyfile(Path('scripts')/name,copies/name)
 files=[Path('scripts')/n for n in names]+[R/'experiment-plan.json',R/'all-finals-before-inference.json',R/'mixed49-plan-TRAIN-s49-job.json',R/'mixed49-plan-TRAIN-s49.log',R/'mixed49-plan-TRAIN-s49/predictions.jsonl',R/'mixed49-plan-TRAIN-s49/before-execution.json']+[f for f in original.iterdir() if f.is_file()]
 write(S/'operation-amendment.json',{'registered_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'purpose':'Complete original frozen R034 comparison after brace-formatting logger exception; no scientific recipe changes','original_plan_SHA256':sha(R/'experiment-plan.json'),'original_controller_failed_exit_preserved':True,'CPU_replay_original_frozen_predictions':48,'CPU_execution_full_trials':96,'CPU_source_parity_full_trials':96,'new_model_predictions_for_recovery':0,'new_learning_steps':0,'unstarted_original_labels':pending,'original_ALL_rules_AST_identical':True,'literal_log_patch_only_base_agent_exc_info_error_call':True,'no_tool_argument_action_parser_prompt_model_scorer_or_rule_change':True,'candidate_tool_error_remains_ERROR_with_skipped_proposals':True,'registered_runtime_unchanged_except_non_crashing_error_logging':True,'original_GPU_exit1_not_relabelled_exit0':True,'CPU_recovery_tool_timing_new_original_LLM_latency_retained':True,'training_or_completed_GPU_jobs_not_restarted':True,'sha256':{str(f):sha(f) for f in files}})
 print({'registered_operation_SHA256':sha(S/'operation-amendment.json'),'unstarted_original_jobs':len(pending)})
if __name__=='__main__':main()
