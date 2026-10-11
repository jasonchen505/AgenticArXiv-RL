import os,sys,json,collections,datetime
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',STORE_BACKEND='memory',FIGURE_ANALYSIS_BACKEND='extractive');sys.path[:0]=['AgenticArxiv','scripts']
from round034_core import R,read,rows,write,verify,sha
S=R.parent/'round-035-2026-10-10'
from round033_plan_agent import parse_plan
from arm_blind_observations import ArtifactAddressMap
from tools.bootstrap import require_all_tools
from benchmark.semantic_oracle import attach_expected_paper_ids
from benchmark.metrics import extract_metrics,is_strict_success
def collect(full=True):
 require_all_tools('R034 full original-scorer closure');p=read(R/'experiment-plan.json');verify(p);freeze=read(R/'all-finals-before-inference.json')
 for f,h in freeze['sha256'].items():assert sha(f)==h,f
 counts={};hashes={};trial_count=0;closed=0;actual_plan_calls=0
 for label,s in p['eval_jobs'].items():
  job=R/(label+'-job.json')
  if not job.exists():
   assert not full,'Unclosed '+label;continue
  j=read(job);recovered=False
  if label=='mixed49-plan-TRAIN-s49' and j.get('exit_code')==1 and (S/'CPU-recovery-receipt.json').exists():
   receipt=read(S/'CPU-recovery-receipt.json');verify(receipt);assert receipt['original_GPU_exit_code']==1 and receipt['CPU_replayed_full_trials']==96 and receipt['new_model_predictions']==0 and receipt['source96_history_semantic_metrics_prediction_usage_identical'];recovered=True
  if j.get('exit_code')!=0 and not recovered:
   assert not full,'Unclosed '+label;continue
  assert j['gpu_index'] in [4,7] and j['started_at_utc']>freeze['recorded_at_utc'];out=R/label;traces=rows(out/'traces.jsonl');assert len(traces)==s['expected_full_trials'];tasks={w:{t['id']:t for t in attach_expected_paper_ids(read(s['tasks']),p['worlds'][w])} for w in (['trainA','trainB'] if s['route']=='plan' else [s['world']])};expected={(w,tid,t) for w,ts in tasks.items() for tid in ts for t in range(s['repeat'])};assert {(x['world'],x['task_id'],x['trial']) for x in traces}==expected;mapper=ArtifactAddressMap({n:str(Path('data/repro_round034')/label/n) for n in ['raw','translated','figures']});c={w:collections.Counter() for w in tasks}
  if s['route']=='plan':
   predictions=rows(out/'predictions.jsonl');byid={(x['task_id'],x['trial']):x for x in predictions};assert len(predictions)==len(tasks['trainA'])*s['repeat'];assert read(out/'before-execution.json')['sha256']==sha(out/'predictions.jsonl');actual_plan_calls+=len(predictions)
  for x in traces:
   z=x['raw_result'];m=extract_metrics(tasks[x['world']][x['task_id']],z,'single_plan' if s['route']=='plan' else 'regex',x['trial'],x['session_id']);assert m.to_dict()==x['metrics'] and is_strict_success(m)==x['strict'];usage=collections.Counter();calls=z['proposal_trace'];assert len(calls)==z['LLM_calls']<=z['max_LLM_calls']
   for q in calls:assert q['request']['temperature']==.1 and q['request']['max_tokens']==1000 and 'data/repro_round034' not in json.dumps(q['request'],ensure_ascii=False);usage.update(q['response']['usage'])
   assert dict(usage)==z['token_usage'] and usage['prompt_tokens']+usage['completion_tokens']==usage['total_tokens']
   for q in z['artifact_observations']:assert mapper.public(q['actual_result'])==q['public_result']
   if s['route']=='plan':
    assert z['proposal_trace']==[byid[x['task_id'],x['trial']]['prediction']];em=extract_metrics(tasks[x['world']][x['task_id']],dict(z,history=z['execution_history']),'executed_single_plan',x['trial'],x['session_id']);assert em.to_dict()==x['execution_metrics'] and is_strict_success(em)==x['execution_strict']
   else:em=m
   c[x['world']].update(runs=1,primary_strict=int(is_strict_success(m)),execution_strict=int(is_strict_success(em)),FF=int(m.false_finish),FINISH=int(m.task_completed),plan_valid=int(z.get('plan_parse_valid',True)),clipped=sum(q['response']['choices'][0].get('finish_reason')=='length' for q in calls),logical_LLM_calls=len(calls),**dict(usage))
  counts[label]=c;closed+=1;trial_count+=len(traces)
  for f in [job,out/'traces.jsonl',out/'metrics.json']:hashes[str(f)]=sha(f)
  if s['route']=='plan':
   for f in [out/'predictions.jsonl',out/'before-execution.json']:hashes[str(f)]=sha(f)
 if full:assert closed==25 and trial_count==1488 and actual_plan_calls==432
 return {'counts':counts,'closed_inference_jobs':closed,'full_trials_rescored':trial_count,'actual_plan_model_calls':actual_plan_calls,'scoring_drift':0,'evidence_sha256':hashes,'original_GPU_exit0_jobs':sum(read(R/(label+'-job.json')).get('exit_code')==0 for label in counts),'CPU_recovered_GPU_failed_units':int('mixed49-plan-TRAIN-s49' in counts),'original_failed_GPU_job_preserved':True}
def rules(c):
 checks={'TRAIN_plan_fit_strict44_parse46_no_clip':True,'fresh_each_seed_world_gain3_vs_source_native_source_plan_lr0_plan':True,'native_old_and_new_Dev_strict_retention_and_FF':True,'fresh_FF_no_worse_source_native_source_plan_lr0_plan':True,'fresh_logical_tokens_at_most_half_source_native':True,'fresh_plan_valid46_and_no_clip_each_seed_world':True}
 for w in ['trainA','trainB']:
  a=c['mixed49-plan-TRAIN-s49'][w];checks['TRAIN_plan_fit_strict44_parse46_no_clip'] &= a['primary_strict']>=44 and a['plan_valid']>=46 and a['clipped']==0
 for seed in [49,50]:
  for w in ['trainA','trainB']:
   a=c[f'mixed49-plan-confirm-s{seed}'][w];b=c[f'source-plan-confirm-s{seed}'][w];d=c[f'lr0_49-plan-confirm-s{seed}'][w];n=c[f'source-native-confirm{w[-1]}-s{seed}'][w];checks['fresh_each_seed_world_gain3_vs_source_native_source_plan_lr0_plan'] &= a['primary_strict']>=max(b['primary_strict'],d['primary_strict'],n['primary_strict'])+3;checks['fresh_FF_no_worse_source_native_source_plan_lr0_plan'] &= a['FF']<=min(b['FF'],d['FF'],n['FF']);checks['fresh_logical_tokens_at_most_half_source_native'] &= 2*a['total_tokens']<=n['total_tokens'];checks['fresh_plan_valid46_and_no_clip_each_seed_world'] &= a['plan_valid']>=46 and a['clipped']==0
  for g,w in [('olddev','old'),('newdev','trainA')]:
   a=c[f'mixed49-native-{g}-s{seed}'][w];b=c[f'source-native-{g}-s{seed}'][w];d=c[f'lr0_49-native-{g}-s{seed}'][w];checks['native_old_and_new_Dev_strict_retention_and_FF'] &= a['primary_strict']>=max(b['primary_strict'],d['primary_strict']) and a['FF']<=min(b['FF'],d['FF'])
 return checks
def main():
 assert not (S/'study-outcome.json').exists();s=collect();checks=rules(s['counts']);write(S/'study-outcome.json',dict(s,completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rule_checks=checks,registered_ALL_rule_met=all(checks.values()),fixed_recipe='eligible_for_independent_review' if all(checks.values()) else 'closed',source_SFT_remains_default=True,no_automatic_promotion=True,training_seeds=1,generation_seeds=[49,50],limited_two_program_families_not_broad_or_paper_blind=True));print(checks)
if __name__=='__main__':main()
