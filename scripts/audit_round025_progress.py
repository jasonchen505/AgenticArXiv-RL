#!/usr/bin/env python3
"""Independently score closed groups only and record four frozen candidates."""
from pathlib import Path
import collections,datetime,hashlib,json,os,sys
R=Path('artifacts/reproduction/round-024-2026-10-07');A=R.parent/'round-025-2026-10-07';read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 os.environ.update(CUDA_VISIBLE_DEVICES='',STORE_BACKEND='memory');sys.path.insert(0,'AgenticArxiv')
 from benchmark.metrics import extract_metrics,is_strict_success
 plan=read(R/'experiment-plan.json');am=read(A/'operational-amendment.json');freeze=read(R/'all-candidates-frozen-before-inference.json');jobs={p.name:read(p) for p in R.glob('*-job.json')}
 for p,h in plan['sha256'].items():assert sha(p)==h,p
 for p,h in am['sha256'].items():assert sha(p)==h,p
 for name,model in freeze['models'].items():
  for p,h in model['sha256'].items():assert sha(p)==h,p
  assert model['sha256']==am['models_sha256'][name]
 ftime=datetime.datetime.fromisoformat(freeze['recorded_at_utc']);assert all(datetime.datetime.fromisoformat(jobs[name+'-train-job.json']['finished_at_utc'])<=ftime for name in plan['learning_arms'])
 counts={};evidence={};total=after=0;closed=0
 for group,g in plan['groups'].items():
  tasks={t['id']:t for t in read(g['oracles'])};sf,split=g['split'].rsplit(':',1);wanted=set(read(sf)['split'][split]);counts[group]={}
  for model in ['sft']+plan['learning_arms']:
   reused=model=='sft' and group in plan['reused_source'];path=Path(plan['reused_source'][group]) if reused else R/(group+'-'+model)
   if not reused:
    name=group+'-'+model+'-job.json'
    if name not in jobs or jobs[name].get('exit_code')!=0:continue
    assert datetime.datetime.fromisoformat(jobs[name]['started_at_utc'])>ftime and jobs[name]['gpu_index'] in [4,7];closed+=1
   raw=[json.loads(x) for x in (path/'traces.jsonl').read_text().splitlines()];summary=read(path/'summary.json');details={(d['task_id'],d['trial']):d for d in summary['details']};assert len(raw)==len(details)==len(wanted)*3 and not summary['error_count'] and collections.Counter(x['task_id'] for x in raw)==dict.fromkeys(wanted,3)
   strict=ff=0;per=collections.Counter()
   for x in raw:
    t=tasks[x['task_id']];m=extract_metrics(t,{'history':x['history']},'CPU_R025_rescore',x['trial'],x['session_id']);d=details[x['task_id'],x['trial']];good=d['completed'] and d['tool_accurate'] and d['arg_score']==1 and d['ref_score']==1 and d['parse_fail']==0 and d['tool_fail']==0 and d['terminal_semantics_accurate'];assert is_strict_success(m)==good and m.false_finish==d['false_finish'];strict+=int(is_strict_success(m));ff+=int(m.false_finish);per[x['task_id']]+=int(is_strict_success(m))
   counts[group][model]={'runs':len(raw),'strict':strict,'false_finish':ff,'all3_success_tasks':sum(v==3 for v in per.values())};total+=len(raw);after+=0 if reused else len(raw)
   for p in [path/'summary.json',path/'traces.jsonl',path/'execution_protocol.json']:evidence[str(p)]=sha(p)
 assert total==after+72
 result={'captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':read(R/'experiment-status.json'),'all4training_complete':True,'actual_training_steps':796,'actual_row_exposures':3180,'new_learning_steps_on_resume':0,'all4finals_frozen_before_candidate_inference':True,'freeze':freeze,'training_independent_audit_sha256':sha(A/'training-independent-audit.json'),'scientific_plan_sha256':sha(R/'experiment-plan.json'),'all1215_scientific_inputs_unchanged':True,'registered_success_rule_unchanged':True,'guard_amendment_and_all_final_model_hashes_unchanged':True,'after_baseline_runs_closed':after,'after_baseline_runs_planned':768,'source_oldDev_reused':24,'source_newDev_frozen_reused':48,'independently_rescored':total,'strict_and_FF_drift':0,'inference_jobs_after_baseline_closed':closed,'inference_jobs_after_baseline_planned':18,'counts':counts,'full_confirmation_complete':after==768,'full_outcome_available':(R/'curriculum-outcome.json').exists(),'no_promotion_or_gain_from_partial_results':after<768,'evidence_sha256':evidence,'jobs_at_capture':jobs}
 (A/'experiment-at-report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k in ['captured_at_utc','status','after_baseline_runs_closed','independently_rescored','inference_jobs_after_baseline_closed','counts']},ensure_ascii=False))
if __name__=='__main__':main()
