#!/usr/bin/env python3
from pathlib import Path
import collections,datetime,hashlib,json,os,re,sys
R=Path('artifacts/reproduction/round-027-2026-10-08');read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def collect(full=False):
 os.environ.update(CUDA_VISIBLE_DEVICES='',STORE_BACKEND='memory');sys.path.insert(0,'AgenticArxiv')
 from benchmark.metrics import extract_metrics,is_strict_success
 plan=read(R/'experiment-plan.json')
 for p,h in plan['sha256'].items():assert sha(p)==h,p
 freeze=read(R/'all-finals-before-inference.json') if (R/'all-finals-before-inference.json').exists() else None
 if freeze:
  for m in freeze['models'].values():
   for p,h in m['sha256'].items():assert sha(p)==h,p
 counts={};evidence={};per={};total=after=jobsclosed=0
 for group,g in plan['groups'].items():
  tasks={t['id']:t for t in read(g['oracles'])};sf,key=g['split'].rsplit(':',1);wanted=set(read(sf)['split'][key]);counts[group]={};per[group]={}
  for model in plan['models_order']:
   reused=plan['reused_evals'].get(group,{}).get(model);path=Path(reused) if reused else R/(group+'-'+model)
   if not reused:
    jp=R/(group+'-'+model+'-job.json')
    if not jp.exists() or read(jp).get('exit_code')!=0:
     assert not full,'Missing closed '+str(jp);continue
    j=read(jp);assert freeze and datetime.datetime.fromisoformat(j['started_at_utc'])>datetime.datetime.fromisoformat(freeze['recorded_at_utc']) and j['gpu_index'] in [4,7];jobsclosed+=1
   raw=[json.loads(x) for x in (path/'traces.jsonl').read_text().splitlines()];summary=read(path/'summary.json');details={(d['task_id'],d['trial']):d for d in summary['details']};keys={(x['task_id'],x['trial']) for x in raw}
   assert keys==set(details) and len(keys)==len(raw)==len(wanted)*3 and not summary['error_count'] and collections.Counter(x['task_id'] for x in raw)==dict.fromkeys(wanted,3)
   strict=ff=0;task=collections.Counter()
   for x in raw:
    d=details[x['task_id'],x['trial']];m=extract_metrics(tasks[x['task_id']],{'history':x['history']},'CPU_R027',x['trial'],x['session_id']);good=d['completed'] and d['tool_accurate'] and d['arg_score']==1 and d['ref_score']==1 and d['parse_fail']==0 and d['tool_fail']==0 and d['terminal_semantics_accurate'];assert is_strict_success(m)==good and m.false_finish==d['false_finish'];strict+=int(good);ff+=int(m.false_finish);task[x['task_id']]+=int(good)
   counts[group][model]={'runs':len(raw),'strict':strict,'false_finish':ff,'all3_success_tasks':sum(v==3 for v in task.values())};per[group][model]=dict(task);total+=len(raw);after+=0 if reused else len(raw)
   for p in [path/'traces.jsonl',path/'summary.json',path/'execution_protocol.json']:evidence[str(p)]=sha(p)
 training={};logged={}
 for arm in plan['learning_arms']:
  final=Path(plan['model_paths'][arm]);jp=R/(arm+'-train-job.json')
  if jp.exists() and read(jp).get('exit_code')==0:
   m=read(final/'training_manifest.json');assert m['actual_steps']==199 and m['dataset_rows']==795 and m['anchor_forward_backward_calls']==795 and m['anchor_exposures_order']==plan['anchor_orders'][str(m['seed'])] and m['reference_equal_source_tensors']==m['changed_adapter_tensors']==392;training[arm]={'steps':199,'CE_rows':795,'anchor_rows':795,'reference_equal':392,'adapter_changed':392,'job':read(jp),'manifest_sha256':sha(final/'training_manifest.json')}
  log=R/(arm+'-train.log')
  if log.exists():
   steps=[int(x) for x in re.findall(r'(\d+)/199',log.read_text(errors='replace'))];logged[arm]=max(steps,default=0)
 if full:assert total==1512 and after==1152 and jobsclosed==26 and len(training)==4
 return {'captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_only':True,'plan_sha256':sha(R/'experiment-plan.json'),'all_scientific_input_hashes_unchanged':True,'completed_final_training_steps':sum(v['steps'] for v in training.values()),'training_finals':training,'logged_steps':logged,'planned_new_training_steps':796,'new_CE_row_exposures_planned':3180,'new_anchor_row_exposures_planned':3180,'all_new_finals_frozen_before_inference':bool(freeze),'freeze':freeze,'after_baseline_runs_closed':after,'after_baseline_runs_planned':1152,'historical_runs_reused':360,'independently_rescored':total,'strict_and_FF_drift':0,'inference_jobs_closed':jobsclosed,'inference_jobs_planned':26,'full_evaluation_complete':after==1152,'counts':counts,'strict_per_task':per,'evidence_sha256':evidence,'status':read(R/'experiment-status.json') if (R/'experiment-status.json').exists() else {'phase':'registered_not_started'},'no_candidate_gain_claim_from_partial_scores':after<1152}
def main():
 assert not (R/'experiment-at-report.json').exists();s=collect();(R/'experiment-at-report.json').write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:s[k] for k in ['captured_at_utc','completed_final_training_steps','logged_steps','independently_rescored','status']},ensure_ascii=False))
if __name__=='__main__':main()
