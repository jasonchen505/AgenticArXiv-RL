import os,sys,json,datetime,collections
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',STORE_BACKEND='memory',FIGURE_ANALYSIS_BACKEND='extractive');sys.path[:0]=['AgenticArxiv','scripts']
from round031_study_core import read,write,sha
from run_round033_pilot import R,verify
from arm_blind_observations import ArtifactAddressMap
from round033_plan_agent import parse_plan
from tools.bootstrap import require_all_tools
from benchmark.metrics import extract_metrics,is_strict_success
from benchmark.semantic_oracle import attach_expected_paper_ids
def collect(p):
 require_all_tools('R033 full pilot closure');verify(p);tasks={w:{x['id']:x for x in attach_expected_paper_ids(read(p['tasks']),s)} for w,s in p['worlds'].items()};counts={w:{} for w in tasks};hashes={};allrows={};tool_results=0
 for job in p['job_order']:
  jp=R/(job+'-job.json');j=read(jp);assert j['exit_code']==0 and j['gpu_index'] in [4,7] and j['started_at_utc']>p['registered_at_utc'];out=R/job;rows=[json.loads(x) for x in (out/'traces.jsonl').read_text().splitlines()];assert len(rows)==(96 if job=='planner' else 48);assert len({(x['world'],x['task_id']) for x in rows})==len(rows);allrows[job]=rows;root=Path('data/repro_round033')/job;mapper=ArtifactAddressMap({n:str(root/n) for n in ['raw','translated','figures']})
  for x in rows:
   world=x['world'];z=x['raw_result'];m=extract_metrics(tasks[world][x['task_id']],z,'single_plan' if job=='planner' else 'regex',0,x['session_id']);assert m.to_dict()==x['metrics'] and is_strict_success(m)==x['strict'];calls=z['proposal_trace'];assert len(calls)==z['LLM_calls']<=z['max_LLM_calls'];usage=collections.Counter()
   for q in calls:assert q['request']['temperature']==.1 and q['request']['max_tokens']==1000 and 'data/repro_round033' not in json.dumps(q['request'],ensure_ascii=False);usage.update(q['response']['usage'])
   assert dict(usage)==z['token_usage'] and usage['prompt_tokens']+usage['completion_tokens']==usage['total_tokens']
   for y in z['artifact_observations']:assert mapper.public(y['actual_result'])==y['public_result'];tool_results+=1
   em=extract_metrics(tasks[world][x['task_id']],dict(z,history=z['execution_history']),'executed_single_plan',0,x['session_id'])
   if job=='planner':assert em.to_dict()==x['execution_metrics'] and is_strict_success(em)==x['execution_strict'];assert len(calls)==1
   c=counts[world].setdefault('planner' if job=='planner' else 'react',collections.Counter());c.update(runs=1,primary_strict=int(is_strict_success(m)),execution_strict=int(is_strict_success(em)),false_finish=int(m.false_finish),FINISH=int(m.task_completed),parse_valid=int(z.get('plan_parse_valid',True)),tool_attempts=len(z['artifact_observations'])+len(z.get('tool_dispatch_errors',[])),logical_LLM_calls=len(calls),**dict(usage))
  for f in [jp,out/'traces.jsonl',out/'metrics.json']:hashes[str(f)]=sha(f)
 predictions=[json.loads(x) for x in (R/'planner-predictions.jsonl').read_text().splitlines()];assert len(predictions)==48;freeze=read(R/'planner-predictions-before-execution.json');assert freeze['real_LLM_calls']==48 and freeze['sha256']==sha(R/'planner-predictions.jsonl');byid={x['task_id']:x for x in predictions};parsed=0;clipped=0
 for x in predictions:
  q=x['prediction'];choices=q['response']['choices'];clipped+=choices[0].get('finish_reason')=='length'
  try:parse_plan(choices[0]['message']['content']);parsed+=1
  except (ValueError,TypeError):pass
 for x in allrows['planner']:assert x['raw_result']['proposal_trace']==[byid[x['task_id']]['prediction']]
 assert all(counts[w]['react']['runs']==counts[w]['planner']['runs']==48 for w in tasks);gate={'parse_at_least44_of48':parsed>=44,'no_output_token_cap_clipping':clipped==0,'primary_strict_at_least40_of48_each_TRAIN_world':all(counts[w]['planner']['primary_strict']>=40 for w in tasks),'logical_total_tokens_at_most_half_ReAct_each_world':all(2*counts[w]['planner']['total_tokens']<=counts[w]['react']['total_tokens'] for w in tasks)}
 return {'counts':counts,'gate_checks':gate,'eligible_for_separately_registered_next_stage':all(gate.values()),'real_plan_model_calls':48,'logical_plan_execution_runs':96,'new_ReAct_full_trials':96,'full_trial_runs':192,'unique_visible_TRAIN_requests':48,'model_predictions_reused_uniformly_two_worlds':True,'per_logical_plan_run_full_cost_charged':True,'valid_plan_predictions':parsed,'output_token_cap_clipped_predictions':clipped,'native_tool_result_mappings_recomputed':tool_results,'scoring_drift':0,'evidence_sha256':hashes,'new_learning_steps':0,'source_SFT_remains_default':True,'TRAIN_feasibility_only_no_generalization_or_adoption_claim':True,'no_automatic_heldout_or_learning_launch':True,'plan_sha256':sha(R/'pilot-plan.json')}
def main():
 assert not (R/'pilot-outcome.json').exists();p=read(R/'pilot-plan.json');z=collect(p);z.update(completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),fixed_pilot='eligible' if z['eligible_for_separately_registered_next_stage'] else 'closed');write(R/'pilot-outcome.json',z);print(z['gate_checks'])
if __name__=='__main__':main()
