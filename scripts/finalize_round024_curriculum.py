#!/usr/bin/env python3
"""Full 840-trial frozen curriculum outcome; no hidden stopping or promotion."""
from pathlib import Path
import collections,datetime,hashlib,json,os,subprocess,sys
R=Path('artifacts/reproduction/round-024-2026-10-07');read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 assert not (R/'curriculum-outcome.json').exists();plan=read(R/'experiment-plan.json');frozen=read(R/'all-candidates-frozen-before-inference.json')
 for p,h in plan['sha256'].items():assert sha(p)==h,p
 for m in frozen['models'].values():
  for p,h in m['sha256'].items():assert sha(p)==h,p
 jobs=[read(p) for p in R.glob('*-job.json')];assert len(jobs)==23 and all(j['exit_code']==0 and j['gpu_index'] in [4,7] for j in jobs)
 ft=datetime.datetime.fromisoformat(frozen['recorded_at_utc']);assert all(datetime.datetime.fromisoformat(j['finished_at_utc'])<=ft for j in jobs if 'scripts/train_round024_curriculum.py' in j['command'])
 assert all(datetime.datetime.fromisoformat(j['started_at_utc'])>ft for p in R.glob('*-job.json') if p.name!='source-new-dev-job.json' for j in [read(p)] if 'scripts/run_frozen_task_file.py' in j['command'])
 os.environ.update(CUDA_VISIBLE_DEVICES='',STORE_BACKEND='memory');sys.path.insert(0,'AgenticArxiv')
 from benchmark.metrics import extract_metrics,is_strict_success
 counts={};evidence={};n=0;comparisons=[]
 for group,g in plan['groups'].items():
  counts[group]={};tasks={t['id']:t for t in read(g['oracles'])};f,split=g['split'].rsplit(':',1);wanted=set(read(f)['split'][split])
  for model in ['sft']+plan['learning_arms']:
   reused=group in plan['reused_source'] and model=='sft';path=Path(plan['reused_source'][group]) if reused else R/(group+'-'+model)
   raw=[json.loads(x) for x in (path/'traces.jsonl').read_text().splitlines()];summary=read(path/'summary.json');details={(d['task_id'],d['trial']):d for d in summary['details']};assert not summary['error_count'] and len(raw)==len(details)==len(wanted)*3 and collections.Counter(x['task_id'] for x in raw)==dict.fromkeys(wanted,3)
   strict=ff=0;per=collections.Counter()
   for x in raw:
    m=extract_metrics(tasks[x['task_id']],{'history':x['history']},'CPU_full_curriculum',x['trial'],x['session_id']);d=details[x['task_id'],x['trial']];good=d['completed'] and d['tool_accurate'] and d['arg_score']==1 and d['ref_score']==1 and d['parse_fail']==0 and d['tool_fail']==0 and d['terminal_semantics_accurate'];assert is_strict_success(m)==good and m.false_finish==d['false_finish'];strict+=int(is_strict_success(m));ff+=int(m.false_finish);per[x['task_id']]+=int(is_strict_success(m))
   counts[group][model]={'runs':len(raw),'strict':strict,'false_finish':ff,'all3_success_tasks':sum(v==3 for v in per.values())};n+=len(raw)
   for p in [path/'traces.jsonl',path/'summary.json',path/'execution_protocol.json']:evidence[str(p)]=sha(p)
  for seed in [43,44]:
   for control in ['sft','repeat'+str(seed)]:
    base=plan['reused_source'][group] if control=='sft' and group in plan['reused_source'] else str(R/(group+'-'+control));target=R/(control+'-vs-curriculum'+str(seed)+'-'+group+'.json')
    subprocess.run(['.venv-repro/bin/python','scripts/compare_repro_evals.py','--base',base,'--sft',str(R/(group+'-curriculum'+str(seed))),'--oracles',g['oracles'],'--output',str(target),'--base_label',control,'--sft_label','curriculum'+str(seed),'--training_description','Train-only compositional curriculum vs equal795row199step/action-marginal original replay; full ordered programs disjoint','--evaluation_split',split],check=True)
    assert all(v['authoritative_rescore_drift']==0 for v in read(target)['models'].values());comparisons.append(str(target))
 assert n==840 and len(comparisons)==16
 checks={'both_seeds_each_confirmation_gain_at_least3':all(counts[w]['curriculum'+str(s)]['strict']>=counts[w]['repeat'+str(s)]['strict']+3 for w in ['confirmA','confirmB'] for s in [43,44]),'oldDev_strict_retained':all(counts['olddev']['curriculum'+str(s)]['strict']>=max(10,counts['olddev']['repeat'+str(s)]['strict']) for s in [43,44]),'newDev_strict_retained':all(counts['newdev']['curriculum'+str(s)]['strict']>=max(counts['newdev']['sft']['strict'],counts['newdev']['repeat'+str(s)]['strict']) for s in [43,44]),'oldDev_FF_retained':all(counts['olddev']['curriculum'+str(s)]['false_finish']<=min(6,counts['olddev']['repeat'+str(s)]['false_finish']) for s in [43,44]),'newDev_and_confirmation_FF_retained':all(counts[g]['curriculum'+str(s)]['false_finish']<=counts[g]['repeat'+str(s)]['false_finish'] for g in ['newdev','confirmA','confirmB'] for s in [43,44])}
 assert set(checks)==set(plan['success_rule']) and all(plan['success_rule'].values())
 success=all(checks.values());write(R/'curriculum-outcome.json',{'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'training_steps':796,'row_exposures':3180,'new_after_baseline_runs':768,'source_new_dev_runs':48,'historical_source_Dev_reused':24,'independently_rescored':840,'strict_and_FF_drift':0,'successful_jobs':23,'comparisons':comparisons,'counts':counts,'rule_checks':checks,'registered_rule_met':success,'fixed_curriculum_recipe':'confirmation_passed_keep_unpromoted' if success else 'closed','no_automatic_promotion':True,'scope':plan['scope'],'evidence_sha256':evidence})
 write(R/'experiment-status.json',{'phase':'completed','updated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'training_steps':796,'success_rule_met':success})
if __name__=='__main__':main()
