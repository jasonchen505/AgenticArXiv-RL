#!/usr/bin/env python3
"""Read-only independent closure of retention study; preserve Round027 report snapshots."""
from pathlib import Path
import collections,datetime,hashlib,json,os,sys
os.environ['CUDA_VISIBLE_DEVICES']='';sys.path[:0]=['scripts','AgenticArxiv']
from audit_round027_status import collect,read,sha,R as OLD
R=OLD.parent/'round-028-2026-10-09'
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 R.mkdir(exist_ok=True);assert not (R/'closure-independent-audit.json').exists();s=collect(full=True);plan=read(OLD/'experiment-plan.json');o=read(OLD/'retention-outcome.json');assert s['counts']==o['counts'] and s['evidence_sha256']==o['evidence_sha256']
 import torch
 from safetensors.torch import load_file
 source=load_file(plan['source_model']+'/adapter_model.safetensors',device='cpu');weights={}
 for arm in plan['learning_arms']:
  d=Path(plan['model_paths'][arm]);a=load_file(str(d/'adapter_model.safetensors'),device='cpu');ref=load_file(str(d/'ref/adapter_model.safetensors'),device='cpu');assert set(source)==set(a)==set(ref) and len(a)==392
  changed=sum(not torch.equal(source[k],a[k]) for k in a);equal=sum(torch.equal(source[k],ref[k]) for k in ref);assert changed==equal==392
  m=read(d/'training_manifest.json');logs=m['anchor_logs'];assert len(logs)==795 and [x['anchor_index'] for x in logs]==plan['anchor_orders'][str(m['seed'])] and all(x['accumulation_group_size']==4 for x in logs[:-3]) and all(x['accumulation_group_size']==3 for x in logs[-3:])
  ks=[x['forward_kl'] for x in logs];weights[arm]={'adapter_changed':changed,'reference_equal_source':equal,'steps':m['actual_steps'],'anchor_exposures':795,'unique_anchors':len({x['anchor_index'] for x in logs}),'mean_anchor_KL':sum(ks)/len(ks),'max_anchor_KL':max(ks),'CE_data_sha256':m['data_sha256']}
 comparisons={}
 for p in o['comparisons']:
  v=read(p);group=Path(p).stem.rsplit('-',1)[1];labels=list(v['models']);base,target=labels;old=s['strict_per_task'][group][base];new=s['strict_per_task'][group][target]
  assert set(old)==set(new);gains=sum(old[k]<3 and new[k]==3 for k in old);losses=sum(old[k]==3 and new[k]<3 for k in old);assert gains==v['task_gains'] and losses==v['task_losses']
  for name,d in v['models'].items():
   c=s['counts'][group][name];assert d['runs']==c['runs'] and d['strict_success_rate']==c['strict']/c['runs'] and d['false_finish_rate']==c['false_finish']/c['runs'] and d['reliability']['pass^3']==c['all3_success_tasks']/(c['runs']//3) and d['authoritative_rescore_drift']==0
  comparisons[p]={'sha256':sha(p),'task_gains':gains,'task_losses':losses}
 assert len(comparisons)==40
 c=s['counts'];each=lambda f:all(f(str(seed)) for seed in [43,44])
 checks={'each_seed_each_fresh_world_course_gain_at_least3_vs_KL_repeat':each(lambda t:all(c[w]['klcurriculum'+t]['strict']>=c[w]['klrepeat'+t]['strict']+3 for w in ['freshA','freshB'])),'oldDev_strict_at_least_source_and_KL_repeat':each(lambda t:c['olddev']['klcurriculum'+t]['strict']>=max(10,c['olddev']['klrepeat'+t]['strict'])),'oldDev_FF_no_worse_than_source_and_KL_repeat':each(lambda t:c['olddev']['klcurriculum'+t]['false_finish']<=min(6,c['olddev']['klrepeat'+t]['false_finish'])),'newDev_strict_retained_vs_plain_course_and_KL_repeat':each(lambda t:c['newdev']['klcurriculum'+t]['strict']>=max(c['newdev']['plaincurriculum'+t]['strict'],c['newdev']['klrepeat'+t]['strict'])),'newDev_FF_no_worse_than_plain_course_and_KL_repeat':each(lambda t:c['newdev']['klcurriculum'+t]['false_finish']<=min(c['newdev']['plaincurriculum'+t]['false_finish'],c['newdev']['klrepeat'+t]['false_finish'])),'fresh_strict_retained_vs_plain_course':each(lambda t:all(c[w]['klcurriculum'+t]['strict']>=c[w]['plaincurriculum'+t]['strict'] for w in ['freshA','freshB'])),'fresh_FF_no_worse_than_KL_repeat':each(lambda t:all(c[w]['klcurriculum'+t]['false_finish']<=c[w]['klrepeat'+t]['false_finish'] for w in ['freshA','freshB']))}
 assert checks==o['rule_checks'] and not all(checks.values()) and not o['registered_rule_met'] and o['fixed_recipe']=='closed'
 jobs=[read(OLD/(name+'-train-job.json')) for name in plan['learning_arms']]+[read(OLD/(g+'-'+m+'-job.json')) for g in plan['groups'] for m in plan['models_order'] if m not in plan['reused_evals'].get(g,{})];assert len(jobs)==30 and all(j['exit_code']==0 and j['gpu_index'] in [4,7] for j in jobs)
 result=dict(s,checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,actual_new_training_steps=796,weights=weights,comparisons_verified=40,comparisons= comparisons,rule_checks=checks,registered_rule_met=False,fixed_recipe_closed=True,no_promotion=True,successful_training_inference_jobs=30,successful_preflight_jobs=1,failed_preflight_attempts_preserved=1,scientific_inputs_verified=len(plan['sha256']),new_learning_steps_this_round=0,outcome_sha256=sha(OLD/'retention-outcome.json'))
 write(R/'closure-independent-audit.json',result);write(R/'retention-outcome.json',o);print(json.dumps({k:result[k] for k in ['passed','independently_rescored','comparisons_verified','actual_new_training_steps','rule_checks']},ensure_ascii=False))
if __name__=='__main__':main()
