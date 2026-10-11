#!/usr/bin/env python3
"""Complete780trials independent scoring and fixed world-data stopping rule."""
from pathlib import Path
import collections,datetime,hashlib,json,os,subprocess,sys
R=Path('artifacts/reproduction/round-023-2026-10-07');read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def good(d):return d['completed'] and d['tool_accurate'] and d['arg_score']==1 and d['ref_score']==1 and d['parse_fail']==0 and d['tool_fail']==0 and d['terminal_semantics_accurate']
def main():
    assert not (R/'world-data-outcome.json').exists();plan=read(R/'experiment-plan.json');freeze=read(R/'all-candidates-frozen-before-inference.json')
    for p,h in plan['sha256'].items():assert sha(p)==h,p
    for model in freeze['models'].values():
        for p,h in model['sha256'].items():assert sha(p)==h,p
    jobs=[read(p) for p in R.glob('*-job.json')];assert len(jobs)==28 and all(j['exit_code']==0 and j['gpu_index'] in [4,7] for j in jobs)
    frozen_time=datetime.datetime.fromisoformat(freeze['recorded_at_utc']);assert all(datetime.datetime.fromisoformat(j['finished_at_utc'])<=frozen_time for j in jobs if 'scripts/train_round023_worlds.py' in j['command'])
    assert all(datetime.datetime.fromisoformat(j['started_at_utc'])>frozen_time for j in jobs if 'scripts/run_frozen_task_file.py' in j['command'])
    os.environ.update(CUDA_VISIBLE_DEVICES='',STORE_BACKEND='memory');sys.path.insert(0,'AgenticArxiv')
    from benchmark.metrics import extract_metrics,is_strict_success
    counts={};detail={};n=new=0;evidence={};comparisons=[]
    for group,g in plan['groups'].items():
        counts[group]={};detail[group]={};tasks={t['id']:t for t in read(g['oracles'])};splitfile,splitgroup=g['split'].rsplit(':',1);wanted=set(read(splitfile)['split'][splitgroup])
        for model in ['sft','repeat43','repeat44','world43','world44']:
            path=Path(plan['reused_source_Dev']) if group=='dev' and model=='sft' else R/(group+'-'+model)
            raw=[json.loads(x) for x in (path/'traces.jsonl').read_text().splitlines()];summary=read(path/'summary.json');ds={(d['task_id'],d['trial']):d for d in summary['details']}
            assert summary['error_count']==0 and len(raw)==len(ds)==len(wanted)*3 and collections.Counter(x['task_id'] for x in raw)==dict.fromkeys(wanted,3)
            strict=ff=0;per=collections.Counter();perff=collections.Counter()
            for x in raw:
                m=extract_metrics(tasks[x['task_id']],{'history':x['history']},x['agent_type'],x['trial'],x['session_id']);d=ds[(x['task_id'],x['trial'])]
                assert is_strict_success(m)==good(d) and m.false_finish==d['false_finish'];strict+=int(is_strict_success(m));ff+=int(m.false_finish);per[x['task_id']]+=int(is_strict_success(m));perff[x['task_id']]+=int(m.false_finish)
            counts[group][model]={'runs':len(raw),'strict':strict,'false_finish':ff,'all3_success_tasks':sum(v==3 for v in per.values()),'tasks':len(wanted)}
            detail[group][model]={t:{'strict':per[t],'false_finish':perff[t]} for t in wanted};n+=len(raw);new+=0 if group=='dev' and model=='sft' else len(raw)
            for f in [path/'traces.jsonl',path/'summary.json',path/'execution_protocol.json']:evidence[str(f)]=sha(f)
        for seed in [43,44]:
            for control in ['sft','repeat'+str(seed)]:
                target=R/(control+'-vs-world'+str(seed)+'-'+group+'.json');base=plan['reused_source_Dev'] if group=='dev' and control=='sft' else str(R/(group+'-'+control))
                subprocess.run(['.venv-repro/bin/python','scripts/compare_repro_evals.py','--base',base,'--sft',str(R/(group+'-world'+str(seed))),'--oracles',g['oracles'],'--output',str(target),'--base_label',control,'--sft_label','world'+str(seed),'--training_description','Two-seed paper-world diversity vs equal1239row310step repeat-budget chosenCE controls, paper-blind confirmation;same task programs','--evaluation_split',splitgroup],check=True)
                assert all(v['authoritative_rescore_drift']==0 for v in read(target)['models'].values());comparisons.append(str(target))
    assert n==780 and new==756 and len(comparisons)==20
    totals={w:{m:sum(counts[w+'-'+part][m]['strict'] for part in ['iid','ood']) for m in ['sft','repeat43','repeat44','world43','world44']} for w in ['confirmA','confirmB']}
    gains={w:{str(seed):totals[w]['world'+str(seed)]-totals[w]['repeat'+str(seed)] for seed in [43,44]} for w in totals}
    checks={'strict_gains':all(v>=2 for values in gains.values() for v in values.values()),'Dev_strict':all(counts['dev']['world'+str(s)]['strict']>=max(10,counts['dev']['repeat'+str(s)]['strict']) for s in [43,44]),'IID_FF':all(counts[w+'-iid']['world'+str(s)]['false_finish']<=counts[w+'-iid']['repeat'+str(s)]['false_finish'] for w in totals for s in [43,44]),'Dev_FF':all(counts['dev']['world'+str(s)]['false_finish']<=min(6,counts['dev']['repeat'+str(s)]['false_finish']) for s in [43,44])}
    changed=[]
    for g in detail:
        for t in sorted(detail[g]['sft']):
            values={m:detail[g][m][t] for m in detail[g]}
            if any(values['world'+str(s)]!=values['repeat'+str(s)] for s in [43,44]):changed.append({'group':g,'task_id':t,'counts':values})
    success=all(checks.values());write(R/'world-data-outcome.json',{'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'new_runs':756,'reused_runs':24,'independently_rescored':780,'strict_and_FF_drift':0,'training_steps':1240,'row_exposures':4956,'successful_jobs':28,'comparisons':comparisons,'counts':counts,'strict_totals':totals,'matched_repeat_gains':gains,'rule_checks':checks,'registered_rule_met':success,'fixed_world_data_recipe':'confirmation_passed_keep_unpromoted' if success else 'closed','all_changed_tasks':changed,'no_automatic_promotion':True,'evidence_sha256':evidence})
    write(R/'experiment-status.json',{'phase':'completed','updated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'new_runs':756,'training_steps':1240,'success_rule_met':success})
if __name__=='__main__':main()
