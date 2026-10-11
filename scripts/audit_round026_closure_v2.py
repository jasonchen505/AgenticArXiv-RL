#!/usr/bin/env python3
"""Read-only full curriculum closure audit; outputs only in Round026, CPU only."""
from pathlib import Path
import collections,datetime,hashlib,json,os,sys
R=Path('artifacts/reproduction/round-024-2026-10-07');A=R.parent/'round-026-2026-10-08';FIX=R.parent/'round-025-2026-10-07'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 A.mkdir(exist_ok=True);assert not (A/'closure-independent-audit.json').exists()
 os.environ.update(CUDA_VISIBLE_DEVICES='',STORE_BACKEND='memory');sys.path.insert(0,'AgenticArxiv')
 from benchmark.metrics import extract_metrics,is_strict_success
 import torch
 from safetensors.torch import load_file
 plan=read(R/'experiment-plan.json');am=read(FIX/'operational-amendment.json');frozen=read(R/'all-candidates-frozen-before-inference.json');outcome=read(R/'curriculum-outcome.json')
 for p,h in dict(plan['sha256'],**am['sha256']).items():assert sha(p)==h,p
 source=load_file('outputs/repro_20261003/sft_e1/final/adapter_model.safetensors',device='cpu');weights={}
 for name,model in frozen['models'].items():
  for p,h in model['sha256'].items():assert sha(p)==h,p
  assert model['sha256']==am['models_sha256'][name]
  d=Path('outputs/repro_round024')/name/'final';m=read(d/'training_manifest.json');a=load_file(str(d/'adapter_model.safetensors'),device='cpu');ref=load_file(str(d/'ref/adapter_model.safetensors'),device='cpu')
  assert set(a)==set(ref)==set(source) and len(a)==392
  changed=sum(not torch.equal(a[k],source[k]) for k in a);same=sum(torch.equal(ref[k],source[k]) for k in ref);assert changed==same==392
  assert m['dataset_rows']==795 and m['actual_steps']==199 and m['dpo_loss_type']==['sft'] and m['seed']==int(name[-2:])
  weights[name]={'rows':795,'steps':199,'adapter_changed':changed,'reference_equal':same}
 jobs={p.name:read(p) for p in R.glob('*-job.json')};assert len(jobs)==23 and all(j['exit_code']==0 and j['gpu_index'] in [4,7] for j in jobs.values())
 ft=datetime.datetime.fromisoformat(frozen['recorded_at_utc'])
 for name,j in jobs.items():
  if name.endswith('-train-job.json'):assert datetime.datetime.fromisoformat(j['finished_at_utc'])<=ft and j['gpu_index']==7
  elif name!='source-new-dev-job.json':assert datetime.datetime.fromisoformat(j['started_at_utc'])>ft
 counts={};evidence={};n=0;per_task={}
 for group,g in plan['groups'].items():
  tasks={t['id']:t for t in read(g['oracles'])};sf,split=g['split'].rsplit(':',1);wanted=set(read(sf)['split'][split]);counts[group]={};per_task[group]={}
  for model in ['sft']+plan['learning_arms']:
   path=Path(plan['reused_source'][group]) if model=='sft' and group in plan['reused_source'] else R/(group+'-'+model)
   raw=[json.loads(x) for x in (path/'traces.jsonl').read_text().splitlines()];s=read(path/'summary.json');details={(d['task_id'],d['trial']):d for d in s['details']}
   assert len(raw)==len(details)==len(wanted)*3 and not s['error_count'] and collections.Counter(x['task_id'] for x in raw)==dict.fromkeys(wanted,3)
   per=collections.Counter();strict=ff=0;categories={}
   for x in raw:
    t=tasks[x['task_id']];m=extract_metrics(t,{'history':x['history']},'CPU_R026_independent',x['trial'],x['session_id']);d=details[x['task_id'],x['trial']]
    good=d['completed'] and d['tool_accurate'] and d['arg_score']==1 and d['ref_score']==1 and d['parse_fail']==0 and d['tool_fail']==0 and d['terminal_semantics_accurate'];assert is_strict_success(m)==good and m.false_finish==d['false_finish']
    strict+=int(good);ff+=int(m.false_finish);per[x['task_id']]+=int(good)
    c=categories.setdefault(t.get('category','unknown'),{'runs':0,'strict':0,'false_finish':0});c['runs']+=1;c['strict']+=int(good);c['false_finish']+=int(m.false_finish)
   counts[group][model]={'runs':len(raw),'strict':strict,'false_finish':ff,'all3_success_tasks':sum(v==3 for v in per.values())};per_task[group][model]={'strict_per_task':dict(per),'categories':categories};n+=len(raw)
   for p in [path/'traces.jsonl',path/'summary.json',path/'execution_protocol.json']:evidence[str(p)]=sha(p)
 assert n==840 and counts==outcome['counts'] and evidence==outcome['evidence_sha256']
 comparisons={}
 for group,g in plan['groups'].items():
  for seed in [43,44]:
   target='curriculum'+str(seed)
   for control in ['sft','repeat'+str(seed)]:
    p=R/(control+'-vs-'+target+'-'+group+'.json');c=read(p)
    for name in [control,target]:
     d=counts[group][name];v=c['models'][name]
     assert v['runs']==d['runs'] and v['strict_success_rate']==d['strict']/d['runs'] and v['false_finish_rate']==d['false_finish']/d['runs'] and v['authoritative_rescore_drift']==0
     assert v['reliability']['pass^3']==d['all3_success_tasks']/(d['runs']//3)
    old=per_task[group][control]['strict_per_task'];new=per_task[group][target]['strict_per_task'];gains=sum(old.get(k,0)<3 and new.get(k,0)==3 for k in old);losses=sum(old.get(k,0)==3 and new.get(k,0)<3 for k in old)
    assert gains==c['task_gains'] and losses==c['task_losses'];comparisons[str(p)]={'task_gains':gains,'task_losses':losses,'strict_delta':counts[group][target]['strict']-counts[group][control]['strict'],'sha256':sha(p)}
 checks={'both_seeds_each_confirmation_gain_at_least3':all(counts[w]['curriculum'+str(s)]['strict']>=counts[w]['repeat'+str(s)]['strict']+3 for w in ['confirmA','confirmB'] for s in [43,44]),'oldDev_strict_retained':all(counts['olddev']['curriculum'+str(s)]['strict']>=max(10,counts['olddev']['repeat'+str(s)]['strict']) for s in [43,44]),'newDev_strict_retained':all(counts['newdev']['curriculum'+str(s)]['strict']>=max(counts['newdev']['sft']['strict'],counts['newdev']['repeat'+str(s)]['strict']) for s in [43,44]),'oldDev_FF_retained':all(counts['olddev']['curriculum'+str(s)]['false_finish']<=min(6,counts['olddev']['repeat'+str(s)]['false_finish']) for s in [43,44]),'newDev_and_confirmation_FF_retained':all(counts[g]['curriculum'+str(s)]['false_finish']<=counts[g]['repeat'+str(s)]['false_finish'] for g in ['newdev','confirmA','confirmB'] for s in [43,44])}
 assert checks==outcome['rule_checks'] and set(checks)==set(plan['success_rule']) and not all(checks.values()) and not outcome['registered_rule_met'] and outcome['fixed_curriculum_recipe']=='closed'
 result={'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'CPU_only':True,'new_training_steps_this_round':0,'independently_rescored':n,'strict_and_FF_drift':0,'comparisons_checked':16,'successful_jobs':23,'scientific_inputs_checked':len(plan['sha256']),'all_final_hashes_unchanged':True,'freeze_chronology_verified':True,'training_weights':weights,'actual_training_steps':796,'row_exposures':3180,'counts':counts,'rule_checks':checks,'registered_rule_met':False,'fixed_recipe_closed':True,'no_promotion':True,'comparisons':comparisons,'evidence_sha256':evidence,'plan_sha256':sha(R/'experiment-plan.json'),'outcome_sha256':sha(R/'curriculum-outcome.json'),'category_and_task_diagnostics':per_task,'scope':outcome['scope']}
 write(A/'closure-independent-audit.json',result);write(A/'curriculum-outcome.json',outcome)
 print(json.dumps({k:result[k] for k in ['passed','independently_rescored','comparisons_checked','successful_jobs','rule_checks']},ensure_ascii=False))
if __name__=='__main__':main()
