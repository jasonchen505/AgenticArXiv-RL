import sys,datetime,shutil
from pathlib import Path
sys.path.insert(0,'scripts')
from round034_core import R,read,write,sha
S=R.parent/'round-035-2026-10-10'
assert not (S/'CPU-startup-amendment.json').exists() and not (S/'CPU-recovery-receipt.json').exists()
shutil.copyfile(S/'CPU-recovery-aligned.log',S/'CPU-context-frozen-settings-failure.log')
names=['replay_round035_CPU_worker.py','recover_round035_CPU_startup.py','register_round035_CPU_startup.py']
for n in names:shutil.copyfile(Path('scripts')/n,S/'source-copies'/n)
files=[Path('scripts')/n for n in names]+[S/'CPU-context-amendment.json',S/'CPU-context-frozen-settings-failure.log']
write(S/'CPU-startup-amendment.json',{'registered_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reason':'Settings is frozen; use separate CPU processes with original per-label PDF env before importing config, matching original GPU startup','prior_failed_CPU_fixtures_preserved':True,'no_mutation_of_config_or_original_scientific_sources':True,'source96_parity_required_before_accepting_candidate_CPU_scores':True,'new_predictions_or_learning':0,'sha256':{str(f):sha(f) for f in files}})
print({'CPU_startup_amendment_SHA256':sha(S/'CPU-startup-amendment.json')})
