#!/usr/bin/env python3
"""CPU-only closure recovery: unchanged split labels, scoring, data and model outputs."""
from pathlib import Path
import collections,datetime,hashlib,json,os,shutil,subprocess,sys
R=Path('artifacts/reproduction/round-019-2026-10-06');A=R.parent/'round-021-2026-10-06';PY='.venv-repro/bin/python'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def good(d):return d['completed'] and d['tool_accurate'] and d['arg_score']==1 and d['ref_score']==1 and d['parse_fail']==0 and d['tool_fail']==0 and d['terminal_semantics_accurate']
def main():
    A.mkdir(parents=True,exist_ok=True);assert not (A/'closure-audit.json').exists()
    plan=read(R/'experiment-plan.json');amend=read(R.parent/'round-020-2026-10-06/operational-amendment.json');frozen=read(R/'all-candidates-frozen-before-inference.json')
    for p,h in {**plan['sha256'],**amend['new_source_sha256']}.items():assert sha(p)==h,p
    for model in frozen['models'].values():
        for p,h in model['sha256'].items():assert sha(p)==h,p
    jobs={p.name:read(p) for p in R.glob('*-job.json')};assert len(jobs)==12 and all(x['exit_code']==0 and x['gpu_index'] in [4,7] for x in jobs.values())
    failure=read(R/'experiment-status.json');assert failure['phase']=='failed_or_capacity_timeout' and 'known-iid' in failure['error']
    write(A/'previous-finalization-failure.json',failure);shutil.copyfile(R.parent/'round-020-2026-10-06/controller.log',A/'previous-controller.log')
    write(A/'CPU-closure-recovery-plan.json',{'registered_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reason':'Compare CLI accepts dev/iid_test/ood_test; controller supplied display group known-iid. Map each frozen group to its existing split enum; no scoring/input/trial changes or rerun.','new_GPU_jobs':0,'new_training_steps':0,'original_plan_sha256':sha(R/'experiment-plan.json'),'source_sha256':sha(Path(__file__)),'evidence_sha256':{str(p):sha(p) for p in R.glob('*/traces.jsonl')}})
    write(R/'experiment-status.json',{'phase':'CPU_closure_recovery','updated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'GPU_jobs':0})
    comparisons=[];metrics={}
    with (A/'CPU-finalization.log').open('w') as log:
        for group,g in plan['groups'].items():
            for seed in [43,44]:
                arm='obs'+str(seed)
                for control in ['sft','replay'+str(seed)]:
                    target=R/(control+'-vs-'+arm+'-'+group+'.json')
                    if not target.exists():
                        subprocess.run([PY,'scripts/compare_repro_evals.py','--base',g['controls'][control],'--sft',str(R/(group+'-'+arm)),'--oracles',g['oracles'],'--output',str(target),'--base_label',control,'--sft_label',arm,'--training_description','All-train observation rendering CE,2seeds111steps; existing diagnostic cases and worlds,not blind confirmation','--evaluation_split',g['split'].rsplit(':',1)[1]],check=True,stdout=log,stderr=log)
                    x=read(target);assert all(m['authoritative_rescore_drift']==0 for m in x['models'].values());metrics.setdefault(group,{}).update(x['models']);comparisons.append(str(target))
    os.environ['CUDA_VISIBLE_DEVICES']='';sys.path[:0]=['AgenticArxiv','scripts']
    from benchmark.metrics import extract_metrics,is_strict_success
    from safetensors.torch import load_file
    import torch
    src=load_file('outputs/repro_20261003/sft_e1/final/adapter_model.safetensors');training=[]
    for seed in [43,44]:
        arm='obs'+str(seed);p=Path(frozen['models'][arm]['model']);m=load_file(str(p/'adapter_model.safetensors'));ref=load_file(str(p/'ref/adapter_model.safetensors'))
        assert len(src)==len(m)==len(ref)==392 and set(src)==set(m)==set(ref)
        assert all(torch.equal(src[k],ref[k]) for k in src) and sum(not torch.equal(src[k],m[k]) for k in src)==392
        v=read(p/'training_manifest.json');assert v['actual_steps']==111 and v['dataset_rows']==443 and v['dpo_loss_type']==['sft'];training.append(v)
        assert datetime.datetime.fromisoformat(jobs[arm+'-train-job.json']['finished_at_utc'])<=datetime.datetime.fromisoformat(frozen['recorded_at_utc'])
    for name,j in jobs.items():
        if '-train-' not in name:assert datetime.datetime.fromisoformat(frozen['recorded_at_utc'])<datetime.datetime.fromisoformat(j['started_at_utc'])
    n=newn=0;counts={};detailed={};evidence={}
    for group,g in plan['groups'].items():
        tasks={t['id']:t for t in read(g['oracles'])};wanted=set(read(g['split'].rsplit(':',1)[0])['split'][g['split'].rsplit(':',1)[1]])
        directories={**g['controls'],**{('obs'+str(s)):str(R/(group+'-obs'+str(s))) for s in [43,44]}}
        counts[group]={};detailed[group]={}
        for model,path in directories.items():
            p=Path(path);summary=read(p/'summary.json');raw=[json.loads(x) for x in (p/'traces.jsonl').read_text().splitlines()];ds={(d['task_id'],d['trial']):d for d in summary['details']}
            assert summary['error_count']==0 and len(raw)==len(ds)==len(wanted)*3 and collections.Counter(x['task_id'] for x in raw)==dict.fromkeys(wanted,3)
            strict=ff=0;per=collections.Counter();perff=collections.Counter()
            for x in raw:
                m=extract_metrics(tasks[x['task_id']],{'history':x['history']},x['agent_type'],x['trial'],x['session_id']);d=ds[(x['task_id'],x['trial'])]
                assert is_strict_success(m)==good(d) and m.false_finish==d['false_finish'],(group,model,x['task_id'],x['trial'])
                strict+=int(is_strict_success(m));ff+=int(m.false_finish);per[x['task_id']]+=int(is_strict_success(m));perff[x['task_id']]+=int(m.false_finish)
            assert strict==round(metrics[group][model]['strict_success_rate']*len(raw))
            counts[group][model]={'runs':len(raw),'strict':strict,'false_finish':ff,'all3_success_tasks':sum(v==3 for v in per.values()),'tasks':len(wanted)}
            detailed[group][model]={t:{'strict':per[t],'false_finish':perff[t]} for t in wanted};n+=len(raw);newn+=len(raw) if model.startswith('obs') else 0
            for f in [p/'traces.jsonl',p/'summary.json',p/'execution_protocol.json']:evidence[str(f)]=sha(f)
    assert n==780 and newn==312 and len(comparisons)==20
    totals={world:{m:sum(counts[world+'-'+part][m]['strict'] for part in ['iid','ood']) for m in ['sft','replay43','replay44','obs43','obs44']} for world in ['known','transfer']}
    effects={w:{str(s):v['obs'+str(s)]-v['replay'+str(s)] for s in [43,44]} for w,v in totals.items()}
    dev={str(s):counts['dev']['obs'+str(s)]['strict'] for s in [43,44]}
    ff_retained=all(counts[g]['obs'+str(s)]['false_finish']<=counts[g]['replay'+str(s)]['false_finish'] for g in ['known-iid','transfer-iid'] for s in [43,44])
    success=all(v>0 for w in effects.values() for v in w.values()) and dev['43']>=11 and dev['44']>=10 and ff_retained
    changed=[]
    for g in detailed:
        for t in sorted(detailed[g]['sft']):
            values={m:detailed[g][m][t] for m in detailed[g]}
            if any(values['obs'+str(s)]!=values['replay'+str(s)] for s in [43,44]):changed.append({'group':g,'task_id':t,'counts':values})
    for p,h in {**plan['sha256'],**amend['new_source_sha256']}.items():assert sha(p)==h,p
    outcome={'new_runs':312,'reused_control_runs':468,'training_steps':222,'metrics':metrics,'counts':counts,'strict_totals':totals,'CE_matched_effects':effects,'Dev_counts':dev,'false_finish_retained':ff_retained,'registered_success_rule_met':success,'no_automatic_promotion':True,'scope':'All-train observation rendering/ephemeral handles post-hoc diagnostic; already observed cases/worlds,not blind confirmation','comparisons':comparisons}
    write(R/'observation-outcome.json',outcome);write(A/'observation-outcome.json',outcome)
    audit={'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':'completed','trials_independently_rescored':780,'new_runs':312,'reused_runs':468,'per_trial_strict_and_false_finish_drift':0,'successful_jobs':12,'training_gpu_indices':[jobs['obs'+str(s)+'-train-job.json']['gpu_index'] for s in [43,44]],'inference_gpu_indices':sorted({j['gpu_index'] for name,j in jobs.items() if '-train-' not in name}),'training_steps':222,'row_exposures':886,'changed_adapter_tensors_each':392,'reference_equal_source_tensors_each':392,'both_candidates_frozen_before_inference':True,'frozen_inputs_unchanged':True,'paired_comparisons':20,'counts':counts,'all_changed_tasks':changed,'strict_totals':totals,'matched_CE_effects':effects,'success_rule_met':success,'evidence_sha256':evidence}
    write(A/'closure-audit.json',audit);write(A/'branch-decision.json',{'registered_rule_met':success,'fixed_observation_recipe':'completed_diagnostic_keep_unpromoted' if success else 'closed','no_coefficient_or_case_tuning':True,'source_reference_retained':True,'both_original_CE_candidates_retained':True,'observation_candidates_promoted':False,'reason':{'matched_CE_effects':effects,'Dev':dev,'false_finish_retained':ff_retained},'new_training_steps_this_interaction':0})
    write(R/'experiment-status.json',{'phase':'completed','updated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'new_runs':312,'training_steps':222,'CPU_recovery_only':True})
    print(json.dumps({'matched_CE_effects':effects,'Dev':dev,'strict_totals':totals,'ff_retained':ff_retained,'rule_met':success,'changed_tasks':len(changed),'independent_trials':n}))
if __name__=='__main__':main()

