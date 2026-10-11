#!/usr/bin/env python3
"""Independently rescore the whole 198-trial transfer, retaining both seeds."""
from pathlib import Path
import collections,datetime,hashlib,json,sys
sys.path[:0]=['scripts','AgenticArxiv']
from benchmark.metrics import extract_metrics,is_strict_success
R=Path('artifacts/reproduction/round-017-2026-10-06');A=Path('artifacts/reproduction/round-018-2026-10-06')
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def good(d):return d['completed'] and d['tool_accurate'] and d['arg_score']==1 and d['ref_score']==1 and d['parse_fail']==0 and d['tool_fail']==0 and d['terminal_semantics_accurate']
def main():
    assert read(R/'transfer-status.json')['phase']=='completed'
    plan=read(R/'cross-world-experiment-plan.json');out=read(R/'transfer-outcome.json');oracles={t['id']:t for t in read(R/'world-oracles.json')}
    for p,h in plan['sha256'].items():assert sha(p)==h,p
    n=0;metrics={};counts={};evidence={};jobs=[]
    for model in ['sft','replay43','replay44']:
        metrics[model]={};counts[model]={}
        for group in ['iid_test','ood_test']:
            path=R/('transfer-'+model+'-'+group);summary=read(path/'summary.json')
            assert summary['error_count']==0
            raw=[json.loads(x) for x in (path/'traces.jsonl').read_text().splitlines()]
            details={(x['task_id'],x['trial']):x for x in summary['details']}
            wanted=set(read(R/'world-splits.json')['split'][group])
            assert len(raw)==len(details)==len(wanted)*3 and {x['task_id'] for x in raw}==wanted
            assert collections.Counter(x['task_id'] for x in raw)==dict.fromkeys(wanted,3)
            strict=ff=0;per=collections.Counter();perff=collections.Counter()
            for x in raw:
                m=extract_metrics(oracles[x['task_id']],{'history':x['history']},x['agent_type'],x['trial'],x['session_id'])
                d=details[(x['task_id'],x['trial'])]
                assert is_strict_success(m)==good(d) and m.false_finish==d['false_finish']
                strict+=int(is_strict_success(m));ff+=int(m.false_finish);per[x['task_id']]+=int(is_strict_success(m));perff[x['task_id']]+=int(m.false_finish)
            assert strict==round(out['metrics'][model][group]['runs']*out['metrics'][model][group]['strict_success_rate'])
            metrics[model][group]={'runs':len(raw),'tasks':len(wanted),'strict_successes':strict,'false_finish_count':ff,'all3_success_tasks':sum(v==3 for v in per.values())}
            counts[model].update({t:{'strict':per[t],'false_finish':perff[t]} for t in wanted});n+=len(raw)
            protocol=read(path/'execution_protocol.json');assert protocol['model']==plan['models'][model] and protocol['dtype']=='bfloat16' and protocol['seed']==45 and protocol['repeat']==3 and not protocol['weights_updated'] and not protocol['oracle_in_prompt']
            job=read(R/('transfer-'+model+'-'+group+'-job.json'));assert job['exit_code']==0 and job['gpu_index']==4;jobs.append(job)
            for p in [path/'summary.json',path/'traces.jsonl',path/'execution_protocol.json',R/('transfer-'+model+'-'+group+'-job.json')]:evidence[str(p)]=sha(p)
    assert n==198 and len(jobs)==6
    comparisons=[R/('sft-vs-'+m+'-'+g+'.json') for m in ['replay43','replay44'] for g in ['iid_test','ood_test']]
    assert all(read(p)['models'][m]['authoritative_rescore_drift']==0 for p in comparisons for m in read(p)['models'])
    totals={m:sum(v['strict_successes'] for v in groups.values()) for m,groups in metrics.items()}
    changed=[{'task_id':t,'strict_counts':{m:counts[m][t]['strict'] for m in counts},'false_finish_counts':{m:counts[m][t]['false_finish'] for m in counts}} for t in sorted(oracles) if any(counts[m][t]!=counts['sft'][t] for m in ['replay43','replay44'])]
    assert totals==out['strict_totals']
    write(A/'transfer-closure-audit.json',{'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':'completed','trials_independently_rescored':198,'strict_and_false_finish_trial_drift':0,'successful_jobs':6,'paired_comparisons':4,'physical_gpu':4,'new_training_steps':0,'frozen_inputs_unchanged':True,'metrics':metrics,'strict_totals':totals,'CE_minus_source':{m:totals[m]-totals['sft'] for m in ['replay43','replay44']},'both_CE_seeds_beat_source':out['both_seeds_beat_source'],'no_automatic_promotion':True,'oldDev_source10_CE43_11_CE44_9':True,'per_task_counts':counts,'all_changed_tasks':changed,'evidence_sha256':evidence,'scope':out['scope']})
    print(json.dumps({'strict_totals':totals,'changed_tasks':len(changed),'independently_rescored':n}))
if __name__=='__main__':main()

