import sys,datetime,shutil
from pathlib import Path
sys.path.insert(0,'scripts')
from round034_core import R,read,write,sha
S=R.parent/'round-035-2026-10-10'
assert not (S/'CPU-context-amendment.json').exists() and not (S/'CPU-recovery-receipt.json').exists()
shutil.copyfile(S/'CPU-recovery.log',S/'CPU-recovery-first-failure.log')
for n in ['recover_round035_aligned_context.py','register_round035_CPU_context.py','diagnose_round035_CPU_parity.py']:shutil.copyfile(Path('scripts')/n,S/'source-copies'/n)
files=[S/'CPU-recovery-first-failure.log',S/'CPU-first-parity-difference.json']+[Path('scripts')/n for n in ['recover_round035_aligned_context.py','register_round035_CPU_context.py','diagnose_round035_CPU_parity.py']]
write(S/'CPU-context-amendment.json',{'registered_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reason':'CPU replay imports config before per-label env update; align three cached settings paths to the original GPU worker startup env, not normalize stored histories','first_CPU_parity_failure_preserved':True,'first_semantic_metric_differences':0,'first_public_history_differences':read(S/'CPU-first-parity-difference.json')['differing_semantic_or_history_trials'],'expected_source96_public_history_semantic_scores_usage_prediction_parity':True,'model_prompt_action_args_oracle_scorer_tool_output_content_unchanged':True,'GPU_generation_or_learning_steps':0,'sha256':{str(f):sha(f) for f in files}})
print({'CPU_context_amendment_SHA256':sha(S/'CPU-context-amendment.json')})
