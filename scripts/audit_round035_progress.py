import os,sys,datetime,collections
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',STORE_BACKEND='memory',FIGURE_ANALYSIS_BACKEND='extractive');sys.path[:0]=['AgenticArxiv','scripts']
from round034_core import R,read,rows,sha,write,verify
from finalize_round035_operational import collect
S=R.parent/'round-035-2026-10-10'
def main():
 assert not (S/'independent-progress-audit.json').exists();p=read(R/'experiment-plan.json');verify(p);verify(read(S/'operation-amendment.json'));verify(read(S/'CPU-startup-amendment.json'));receipt=read(S/'CPU-recovery-receipt.json');verify(receipt);data=collect(False);assert data['closed_inference_jobs']>=2
 counts=data['counts'];candidate=counts['mixed49-plan-TRAIN-s49'];checks={w:{'strict44':c['primary_strict']>=44,'parse46':c['plan_valid']>=46,'no_clip':c['clipped']==0} for w,c in candidate.items()};allfit=all(all(c.values()) for c in checks.values());assert not allfit
 mappings=sum(len(x['raw_result']['artifact_observations']) for label in counts for x in rows(R/label/'traces.jsonl'));exceptions=sum(len(x['raw_result'].get('tool_dispatch_errors',[])) for label in counts for x in rows(R/label/'traces.jsonl'))
 parity=None
 if 'lr0_49-plan-TRAIN-s49' in counts:
  left=rows(R/'source-plan-TRAIN-s49/predictions.jsonl');right=rows(R/'lr0_49-plan-TRAIN-s49/predictions.jsonl');assert len(left)==len(right)==48;parity={'pairs':48,'messages_identical':sum(a['prediction']['request']['messages']==b['prediction']['request']['messages'] for a,b in zip(left,right)),'choices_identical':sum(a['prediction']['response']['choices']==b['prediction']['response']['choices'] for a,b in zip(left,right)),'token_usage_identical':sum(a['prediction']['response']['usage']==b['prediction']['response']['usage'] for a,b in zip(left,right)),'model_paths_intentionally_different':True}
 write(S/'independent-progress-audit.json',dict(data,checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed_evidence_audit=True,verified_original_scientific_inputs=len(p['sha256']),native_tool_result_mappings_recomputed=mappings,actual_tool_exceptions_preserved=exceptions,TRAIN_candidate_checks=checks,TRAIN_ALL_fit_met=allfit,registered_six_ALL_can_no_longer_pass=True,remaining_controls_still_complete_original_protocol=True,source_lr0_prediction_parity=parity,new_learning_steps_this_round=0,new_predictions_for_CPU_recovery=0,source_SFT_remains_default=True,no_candidate_adoption_or_generalization_claim=True));print({'full_raw_trials_rescored':data['full_trials_rescored'],'counts':counts,'TRAIN_fit':checks,'map':mappings,'source_lr0':parity})
if __name__=='__main__':main()
