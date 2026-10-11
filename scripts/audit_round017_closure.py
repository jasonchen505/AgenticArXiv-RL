#!/usr/bin/env python3
"""Independent per-trial closure of all frozen Round014 evidence; CPU only."""
from pathlib import Path
import collections,datetime,json,sys
sys.path[:0]=['scripts','AgenticArxiv']
from run_round014_experiment import R as OLD,MODELS,ARMS,PAIRS,GROUPS,read,sha,write,leaf
from benchmark.metrics import extract_metrics,is_strict_success
R=OLD.parent/'round-017-2026-10-06'
def good(d):return d['completed'] and d['tool_accurate'] and d['arg_score']==1 and d['ref_score']==1 and d['parse_fail']==0 and d['tool_fail']==0 and d['terminal_semantics_accurate']
def main():
    R.mkdir(exist_ok=True);plan=read(OLD/'experiment-plan.json');out=read(OLD/'mainline-outcome.json');completion=read(OLD/'completion.json')
    for p,h in {**plan['sha256'],**plan['reused_evidence_sha256']}.items():assert sha(p)==h,p
    assert completion['state']=='completed' and completion['actual_new_benchmark_runs']==822 and completion['actual_reused_development_runs']==270
    assert not out['success_rule_met'] and not out['both_seeds_beat_matched_supervised']
    from safetensors.torch import load_file
    import torch
    src=load_file(plan['evaluation_models']['sft']+'/adapter_model.safetensors')
    for a in ARMS:
        f=read(OLD/(a+'-candidate-freeze.json'));assert sha(Path(f['model'])/'adapter_model.safetensors')==f['candidate_sha256']
        m=load_file(f['model']+'/adapter_model.safetensors');ref=load_file(f['model']+'/ref/adapter_model.safetensors')
        assert set(src)==set(m)==set(ref) and len(src)==392 and all(torch.equal(src[k],ref[k]) for k in src)
        assert sum(not torch.equal(src[k],m[k]) for k in src)==392
    evidence={};counts={};n=0
    for group in GROUPS:
        oracle=read('artifacts/reproduction/round-002-2026-10-02/task-oracles.json') if group=='dev' else read((OLD.parent/'round-012-2026-10-04' if group.startswith('previous') else OLD)/'fresh-oracles.json')
        tasks={t['id']:t for t in oracle};counts[group]={}
        for model in MODELS:
            directory=OLD/leaf(model,group);summary=read(directory/'summary.json');assert summary['error_count']==0
            raw=[json.loads(line) for line in (directory/'traces.jsonl').read_text().splitlines()];details={(d['task_id'],d['trial']):d for d in summary['details']}
            assert len(raw)==len(details)==out['metrics'][model][group]['runs']
            success=ff=0;per=collections.Counter()
            for x in raw:
                m=extract_metrics(tasks[x['task_id']],{'history':x['history']},x['agent_type'],x['trial'],x['session_id']);d=details[(x['task_id'],x['trial'])]
                assert is_strict_success(m)==good(d) and m.false_finish==d['false_finish'],(model,group,x['task_id'],x['trial'])
                success+=int(is_strict_success(m));ff+=int(m.false_finish);per[x['task_id']]+=int(is_strict_success(m))
            assert success==out['metrics'][model][group]['strict_successes'] and ff==out['metrics'][model][group]['false_finish_count']
            counts[group][model]=dict(per);n+=len(raw)
            for p in [directory/'summary.json',directory/'traces.jsonl',directory/'execution_protocol.json']:evidence[str(p)]=sha(p)
        for b,a in PAIRS:assert (OLD/(b+'-vs-'+a+'-'+group+'.json')).exists()
    assert n==1092
    jobs=[read(p) for p in OLD.glob('*-job.json')];assert len(jobs)==30 and all(j['exit_code']==0 and j['gpu_index']==4 for j in jobs)
    fresh=out['strict_totals']['fresh'];assert [fresh[m] for m in ['sft','ct43','ct44','replay43','replay44','real43','real44']]==[47,50,51,52,52,45,47]
    changed=[]
    for task in sorted(counts['fresh-iid_test']['sft']):
        vals={m:counts['fresh-iid_test'][m][task] for m in MODELS}
        if any(vals['replay'+str(s)]!=vals['real'+str(s)] for s in [43,44]):changed.append({'task_id':task,'strict_counts':vals})
    value={'recorded_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':'completed','strict_and_false_finish_trial_rescore_drift':0,'independently_rescored_trials':n,'new_runs':822,'reused_runs':270,'completed_paired_comparisons':40,'successful_job_count':30,'all_jobs_physical_gpu4':True,'training_steps':444,'training_row_exposures':1772,'frozen_inputs_unchanged':True,'source_equal_reference_tensors_each':392,'changed_adapter_tensors_each':392,'fresh_strict_totals':fresh,'fresh_preference_minus_matched_ce':{'43':-7,'44':-5},'success_rule_met':False,'decision':'Close current real-error anchored-DPO recipe; keep source reference and both matched CE models as unpromoted candidates. No coefficient sweep or bestseed selection.','all_fresh_matched_preference_changed_tasks':changed,'evidence_sha256':evidence,'metrics':out['metrics']}
    write(R/'closure-audit.json',value);write(R/'branch-decision.json',{'closed':['Round014 real-error sigmoidDPO+CE fixed beta.1/weights1:1'],'source_reference_retained':True,'matched_ce_candidates_retained':['replay43','replay44'],'no_automatic_model_promotion':True,'reason':'Bothpreference seeds worse thanmatched CE oncomplete confirmation; CE+5each vs source butoldDev11/9 vssource10,so no bestseed promotion.','next_question':'Do the two supervised candidates transfer to a paper-disjoint world with unchanged task programs?','new_training_authorized_by_this_decision':False,'case_or_coefficient_tuning':False})
    print(json.dumps({k:value[k] for k in ['state','independently_rescored_trials','completed_paired_comparisons','fresh_strict_totals','fresh_preference_minus_matched_ce','success_rule_met']}))
if __name__=='__main__':main()
