import os,sys,json,collections,datetime
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',STORE_BACKEND='memory',FIGURE_ANALYSIS_BACKEND='extractive');sys.path[:0]=['AgenticArxiv','scripts']
from round036_core import R,read,rows,write,verify,sha
from round033_plan_agent import parse_plan
from arm_blind_observations import ArtifactAddressMap
from tools.bootstrap import require_all_tools
from benchmark.semantic_oracle import attach_expected_paper_ids
from benchmark.metrics import extract_metrics,is_strict_success
def collect(full=True):
 require_all_tools('R036 full TRAIN original-scorer closure');p=read(R/'experiment-plan.json');verify(p);freeze=read(R/'all-finals-before-inference.json')
 for f,h in freeze['sha256'].items():assert sha(f)==h,f
 counts={};hashes={};trial_count=0;closed=0;actual_plan_calls=0
 for label,s in p['eval_jobs'].items():
  job=R/(label+'-job.json')
  if not job.exists() or read(job).get('exit_code')!=0:
   assert not full,'Unclosed '+label;continue
  j=read(job);assert j['gpu_index'] in [4,7] and j['started_at_utc']>freeze['recorded_at_utc'];out=R/label;traces=rows(out/'traces.jsonl');assert len(traces)==s['expected_full_trials'];tasks={w:{t['id']:t for t in attach_expected_paper_ids(read(s['tasks']),p['worlds'][w])} for w in (['trainA','trainB'] if s['route']=='plan' else [s['world']])};expected={(w,tid,t) for w,ts in tasks.items() for tid in ts for t in range(s['repeat'])};assert {(x['world'],x['task_id'],x['trial']) for x in traces}==expected;mapper=ArtifactAddressMap({n:str(Path('data/repro_round036')/label/n) for n in ['raw','translated','figures']});c={w:collections.Counter() for w in tasks}
  if s['route']=='plan':
   predictions=rows(out/'predictions.jsonl');byid={(x['task_id'],x['trial']):x for x in predictions};assert len(predictions)==len(tasks['trainA'])*s['repeat'];assert read(out/'before-execution.json')['sha256']==sha(out/'predictions.jsonl');actual_plan_calls+=len(predictions)
  for x in traces:
   z=x['raw_result'];m=extract_metrics(tasks[x['world']][x['task_id']],z,'single_plan' if s['route']=='plan' else 'regex',x['trial'],x['session_id']);assert m.to_dict()==x['metrics'] and is_strict_success(m)==x['strict'];usage=collections.Counter();calls=z['proposal_trace'];assert len(calls)==z['LLM_calls']<=z['max_LLM_calls']
   for q in calls:assert q['request']['temperature']==.1 and q['request']['max_tokens']==1000 and 'data/repro_round036' not in json.dumps(q['request'],ensure_ascii=False);usage.update(q['response']['usage'])
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
 if full:assert closed==3 and trial_count==768 and actual_plan_calls==384
 return {'counts':counts,'closed_inference_jobs':closed,'full_trials_rescored':trial_count,'actual_plan_model_calls':actual_plan_calls,'scoring_drift':0,'evidence_sha256':hashes}
def main():
 assert not (R/'pilot-outcome.json').exists();p=read(R/'pilot-plan.json');assert p==read(R/'experiment-plan.json');a=collect();c=a['counts'];checks={};subgroups={}
 for world in ['trainA','trainB']:
  xs=rows(R/'plan49-plan-TRAIN-s49/traces.jsonl');by={g:[x for x in xs if x['world']==world and x['task_id'].startswith('plan036_')==(g=='long80')] for g in ['old48','long80']};s={g:{'runs':len(z),'strict':sum(x['strict'] for x in z),'parse':sum(x['raw_result']['plan_parse_valid'] for x in z)} for g,z in by.items()};assert s['old48']['runs']==48 and s['long80']['runs']==80;subgroups[world]=s;v=c['plan49-plan-TRAIN-s49'][world]
  checks[world+':TRAIN_strict118_short44_long74']=v['primary_strict']>=118 and s['old48']['strict']>=44 and s['long80']['strict']>=74
  checks[world+':TRAIN_parse123_short46_long77_clip0']=v['plan_valid']>=123 and s['old48']['parse']>=46 and s['long80']['parse']>=77 and v['clipped']==0
  checks[world+':gain3_over_controls']=v['primary_strict']>=max(c['source-plan-TRAIN-s49'][world]['primary_strict'],c['lr0_49-plan-TRAIN-s49'][world]['primary_strict'])+3
 left=rows(R/'source-plan-TRAIN-s49/predictions.jsonl');right=rows(R/'lr0_49-plan-TRAIN-s49/predictions.jsonl');assert len(left)==len(right)==128;parity=True
 for l,r in zip(left,right):
  assert (l['task_id'],l['trial'])==(r['task_id'],r['trial']);lq=dict(l['prediction']['request']);rq=dict(r['prediction']['request']);lq.pop('model');rq.pop('model');parity &= lq==rq and l['prediction']['response']['choices']==r['prediction']['response']['choices'] and l['prediction']['response']['usage']==r['prediction']['response']['usage']
 for l,r in zip(rows(R/'source-plan-TRAIN-s49/traces.jsonl'),rows(R/'lr0_49-plan-TRAIN-s49/traces.jsonl')):
  parity &= (l['task_id'],l['world'],l['trial'])==(r['task_id'],r['world'],r['trial']) and l['raw_result']['history']==r['raw_result']['history'] and l['strict']==r['strict'] and l['metrics']['false_finish']==r['metrics']['false_finish'] and l['raw_result']['token_usage']==r['raw_result']['token_usage']
 checks['source_lr0_prediction_history_semantic_usage_parity']=parity;checks['native_source_frozen_and_prompt_IDs_equal']=sha(Path(p['source_model'])/'adapter_model.safetensors')==p['route_policy']['source_adapter_SHA256'] and read(R/'all-finals-before-inference.json')['all512_prompt_IDs_equal_source_each_model'];write(R/'pilot-outcome.json',dict(a,completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),subgroups=subgroups,rule_checks=checks,TRAIN_ALL_met=all(checks.values()),fixed_recipe='eligible_to_register_future_heldout_only' if all(checks.values()) else 'closed',source_SFT_remains_default=True,not_generalization_evidence=True,no_reserved_tasks_or_predictions_created=True,no_automatic_promotion=True));print(checks)
if __name__=='__main__':main()
