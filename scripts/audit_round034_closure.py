import os,sys,datetime
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';sys.path[:0]=['AgenticArxiv','scripts']
from round031_study_core import read,write,sha
from finalize_round033_pilot import collect
R=Path('artifacts/reproduction/round-034-2026-10-10');OLD=R.parent/'round-033-2026-10-10'
def main():
 R.mkdir(exist_ok=True);assert not (R/'closure-independent-audit.json').exists();p=read(OLD/'pilot-plan.json');s=collect(p);o=read(OLD/'pilot-outcome.json')
 for k,v in s.items():assert o[k]==v,k
 assert o['fixed_pilot']=='closed' and not o['eligible_for_separately_registered_next_stage'];assert s['full_trial_runs']==192 and s['native_tool_result_mappings_recomputed']==597
 ratios={w:s['counts'][w]['planner']['total_tokens']/s['counts'][w]['react']['total_tokens'] for w in p['worlds']}
 write(R/'closure-independent-audit.json',dict(s,passed=True,checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed_at_utc=o['completed_at_utc'],fixed_pilot_closed=True,plan_token_fraction_of_native=ratios,verified_frozen_inputs=len(p['sha256']),raw_outcome_sha256=sha(OLD/'pilot-outcome.json'),no_new_model_predictions=True));print({'closed192':True,'counts':s['counts'],'ratios':ratios})
if __name__=='__main__':main()
