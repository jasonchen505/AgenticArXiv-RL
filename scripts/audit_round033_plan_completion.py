"""Read-only CPU audit of the first completed arm. No early controller stop."""
import os,sys,json,collections,datetime,subprocess,re,tarfile,shutil
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',STORE_BACKEND='memory',FIGURE_ANALYSIS_BACKEND='extractive');sys.path[:0]=['AgenticArxiv','scripts']
from run_round033_pilot import R,verify
from round031_study_core import read,write,sha
from round033_plan_agent import parse_plan
from arm_blind_observations import ArtifactAddressMap
from tools.bootstrap import require_all_tools
from benchmark.metrics import extract_metrics,is_strict_success
from benchmark.semantic_oracle import attach_expected_paper_ids
def main():
 assert not (R/'planner-independent-audit.json').exists();require_all_tools('R033 first closed arm independent audit');p=read(R/'pilot-plan.json');verify(p);j=read(R/'planner-job.json');assert j['exit_code']==0 and j['gpu_index'] in [4,7] and j['started_at_utc']>p['registered_at_utc'];preds=[json.loads(s) for s in (R/'planner-predictions.jsonl').read_text().splitlines()];assert len(preds)==48;freeze=read(R/'planner-predictions-before-execution.json');assert freeze['sha256']==sha(R/'planner-predictions.jsonl');byid={x['task_id']:x for x in preds};valid=0;clipped=0;parse_errors=collections.Counter()
 for x in preds:
  choice=x['prediction']['response']['choices'][0];clipped+=choice.get('finish_reason')=='length'
  try:parse_plan(choice['message']['content']);valid+=1
  except (ValueError,TypeError) as e:parse_errors[type(e).__name__]+=1
 rows=[json.loads(s) for s in (R/'planner/traces.jsonl').read_text().splitlines()];assert len(rows)==96 and len({(x['world'],x['task_id']) for x in rows})==96;counts={};mapper=ArtifactAddressMap({n:str(Path('data/repro_round033/planner')/n) for n in ['raw','translated','figures']});results=0
 for world,snapshot in p['worlds'].items():
  tasks={x['id']:x for x in attach_expected_paper_ids(read(p['tasks']),snapshot)};c=collections.Counter()
  for x in rows:
   if x['world']!=world:continue
   z=x['raw_result'];assert z['proposal_trace']==[byid[x['task_id']]['prediction']];m=extract_metrics(tasks[x['task_id']],z,'single_plan',0,x['session_id']);em=extract_metrics(tasks[x['task_id']],dict(z,history=z['execution_history']),'executed_single_plan',0,x['session_id']);assert m.to_dict()==x['metrics'] and em.to_dict()==x['execution_metrics'] and is_strict_success(m)==x['strict'] and is_strict_success(em)==x['execution_strict'];assert z['proposal_trace'][0]['response']['usage']==z['token_usage'];assert z['LLM_calls']==z['max_LLM_calls']==1
   for y in z['artifact_observations']:assert mapper.public(y['actual_result'])==y['public_result'];results+=1
   c.update(runs=1,primary_strict=int(is_strict_success(m)),execution_strict=int(is_strict_success(em)),false_finish=int(m.false_finish),parse_valid=int(z['plan_parse_valid']),FINISH=int(m.task_completed),**z['token_usage'])
  assert c['runs']==48 and c['parse_valid']==valid;counts[world]=dict(c)
 failed={'parse44_of48':valid<44,'primary40_of48_each_world':any(c['primary_strict']<40 for c in counts.values())};assert any(failed.values())
 evidence={str(f):sha(f) for f in [R/'planner-job.json',R/'planner-predictions.jsonl',R/'planner-predictions-before-execution.json',R/'planner/traces.jsonl',R/'planner/metrics.json']}
 write(R/'planner-independent-audit.json',{'passed_evidence_audit':True,'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'full_completed_plan_runs_rescored':96,'real_plan_predictions':48,'strict_JSON_valid_predictions':valid,'invalid_predictions':48-valid,'output_cap_clipped_predictions':clipped,'parse_exception_counts':parse_errors,'counts':counts,'actual_public_tool_results_recomputed':results,'scoring_drift':0,'registered_conditions_already_failed':failed,'ALL_feasibility_gate_can_no_longer_pass':True,'remaining_ReAct_controls_continue_original_finite_plan':True,'no_partial_before_after_or_cost_gain_claim':True,'TRAIN_only_no_adoption':True,'new_learning_steps':0,'evidence_sha256':evidence})
 launch=read(R/'pilot-launch.json');jobs={};logs={}
 for label in p['job_order']:
  f=R/(label+'-job.json')
  if f.exists():jobs[label]=read(f);write(R/('execution-snapshot-'+f.name),jobs[label])
  f=R/(label+'.log')
  if f.exists():t=f.read_text();(R/('execution-snapshot-'+f.name)).write_text(t);logs[label]={'completed_plan_predictions':len(re.findall(r'^Plan prediction \d+ /48 ',t,re.M)),'ReAct_trial_starts':len(re.findall(r'^\[\d+/48\] task=',t,re.M))}
 write(R/'pilot-execution-at-report.json',{'captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'launch':launch,'status':read(R/'pilot-status.json'),'controller_alive':subprocess.run(['kill','-0',str(launch['pid'])],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0,'jobs':jobs,'logs':logs,'closed_jobs':sum(x.get('exit_code')==0 for x in jobs.values()),'registered_jobs':3,'full_outcome_present':(R/'pilot-outcome.json').exists(),'TRAIN_feasibility_only':True,'new_learning_steps':0,'no_partial_gain_claim':True});shutil.copyfile(Path('scripts/audit_round033_plan_completion.py'),R/'source-copies/audit_round033_plan_completion.py')
 files=[R/'planner-independent-audit.json',R/'pilot-execution-at-report.json',R/'planner-job.json',R/'planner-predictions.jsonl',R/'planner-predictions-before-execution.json',R/'source-copies/audit_round033_plan_completion.py']+list((R/'planner').rglob('*'))+list(R.glob('execution-snapshot-*'));archive=R.parent/'round033-planner-closed.tar.gz';assert not archive.exists()
 with tarfile.open(archive,'w:gz') as tar:
  for f in files:
   if f.is_file():tar.add(f,arcname=str(f.relative_to(R.parent)),recursive=False)
 print(json.dumps({'valid_plans':valid,'counts':counts,'archive_sha256':sha(archive)},ensure_ascii=False))
if __name__=='__main__':main()
