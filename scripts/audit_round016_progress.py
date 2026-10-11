#!/usr/bin/env python3
"""Audit completed training/development without closing unfinished confirmation."""
from pathlib import Path
from datetime import datetime,timezone
import collections,json,subprocess
from run_round014_experiment import R as OLD,ARMS,CONTROLS,MODELS,PAIRS,read,sha,write,leaf
R=OLD.parent/'round-016-2026-10-06'
def good(d):return d['completed'] and d['tool_accurate'] and d['arg_score']==1 and d['ref_score']==1 and d['parse_fail']==0 and d['tool_fail']==0 and d['terminal_semantics_accurate']
def main():
    R.mkdir(exist_ok=True)
    plan=read(OLD/'experiment-plan.json')
    for f,h in {**plan['sha256'],**plan['reused_evidence_sha256']}.items():assert sha(f)==h,f
    from safetensors.torch import load_file
    import torch
    source=load_file(plan['evaluation_models']['sft']+'/adapter_model.safetensors');training={}
    first_eval=min(datetime.fromisoformat(read(p)['started_at_utc']).timestamp() for p in OLD.glob('*-job.json') if 'train-job' not in p.name)
    for a in ARMS:
        m=read(OLD/(a+'-training-manifest.json'));f=read(OLD/(a+'-candidate-freeze.json'));j=read(OLD/(a+'-train-job.json'))
        assert m['actual_steps']==111 and m['dataset_rows']==443 and m['source_adapter_sha256']==plan['source_adapter_sha256'] and m['preference_data_sha256']==sha('data/repro_round014/certified_combined.jsonl')
        assert j['exit_code']==0 and j['gpu_index']==4 and m['reference_equal_sft_and_frozen_tensors']==392
        assert m['dpo_loss_type']==(['sigmoid','sft'] if a.startswith('real') else ['sft']) and m['dpo_loss_weights']==([1.,1.] if a.startswith('real') else [1.])
        model=load_file(f['model']+'/adapter_model.safetensors');ref=load_file(f['model']+'/ref/adapter_model.safetensors')
        assert set(source)==set(model)==set(ref) and sha(Path(f['model'])/'adapter_model.safetensors')==f['candidate_sha256']
        assert all(torch.equal(source[k],ref[k]) for k in source) and sum(not torch.equal(source[k],model[k]) for k in source)==392
        assert datetime.fromisoformat(j['finished_at_utc']).timestamp()<=datetime.fromisoformat(f['frozen_at_utc']).timestamp()<=first_eval
        training[a]={k:m[k] for k in ['objective','seed','actual_steps','dataset_rows','train_loss','torch_peak_gib','dpo_loss_type','dpo_loss_weights']};training[a].update(process_seconds=j['elapsed_seconds'],source_equal_frozen_reference_tensors=392,changed_tensors=392,candidate_sha256=f['candidate_sha256'],before_any_round014_inference=True)
    metrics={a:{} for a in MODELS};counts={};changed=[]
    for g in ['dev','previous-iid_test','previous-ood_test']:
        rescored={}
        for b,a in PAIRS:
            c=read(OLD/(b+'-vs-'+a+'-'+g+'.json'))['models']
            for model,m in c.items():
                assert m['authoritative_rescore_drift']==0
                if model in rescored:assert rescored[model]==m
                rescored[model]=m
        assert set(rescored)==set(MODELS)
        for a in MODELS:
            s=read(OLD/leaf(a,g)/'summary.json');ds=s['details'];assert s['error_count']==0 and len(ds)==(24 if g=='dev' else 54 if g.endswith('iid_test') else 12)
            n=sum(good(d) for d in ds);assert n==round(rescored[a]['strict_success_rate']*len(ds))
            metrics[a][g]={**rescored[a],'strict_successes':n,'false_finish_count':sum(d['false_finish'] for d in ds)}
            counters=collections.Counter()
            for d in ds:counters[d['task_id']]+=int(good(d))
            counts.setdefault(g,{})[a]=dict(counters)
        for task in counts[g]['sft']:
            scores={a:counts[g][a][task] for a in MODELS}
            if len(set(scores.values()))==1:continue
            changed.append({'group':g,'task_id':task,'strict_counts':scores,'raw_trials':{a:[json.loads(s) for s in (OLD/leaf(a,g)/'traces.jsonl').read_text().splitlines() if json.loads(s)['task_id']==task] for a in MODELS}})
    totals={a:sum(metrics[a][g]['strict_successes'] for g in ['previous-iid_test','previous-ood_test']) for a in MODELS}
    effects={str(s):{b:totals['real'+str(s)]-totals[b] for b in ['sft','ct'+str(s),'replay'+str(s)]} for s in [43,44]}
    fresh_sft=read(OLD/'fresh-sft-iid_test/summary.json');assert fresh_sft['sample_count']==54 and fresh_sft['error_count']==0
    # Completed source-only confirmation rows are separately rescored, never padded to66.
    import sys
    sys.path.insert(0,'AgenticArxiv')
    from benchmark.metrics import extract_metrics,is_strict_success
    oracles={t['id']:t for t in read(OLD/'fresh-oracles.json')};source_fresh=[]
    for line in (OLD/'fresh-sft-iid_test/traces.jsonl').read_text().splitlines():
        x=json.loads(line);m=extract_metrics(oracles[x['task_id']],{'history':x['history']},x['agent_type'],x['trial'],x['session_id']);source_fresh.append(is_strict_success(m))
    assert sum(source_fresh)==sum(good(d) for d in fresh_sft['details'])
    done=[];remaining=[];running=[];fresh_metrics={a:{} for a in MODELS}
    for a in MODELS:
        for group in ['fresh-iid_test','fresh-ood_test']:
            label=leaf(a,group);p=OLD/(label+'-job.json')
            if p.exists() and read(p).get('exit_code')==0:
                done.append(label)
                summary=read(OLD/label/'summary.json');assert summary['error_count']==0
                raw=[json.loads(line) for line in (OLD/label/'traces.jsonl').read_text().splitlines()]
                authoritative=[]; per_task=collections.Counter(); details={(d['task_id'],d['trial']):d for d in summary['details']};assert len(details)==len(raw)
                for x in raw:
                    m=extract_metrics(oracles[x['task_id']],{'history':x['history']},x['agent_type'],x['trial'],x['session_id'])
                    d=details[(x['task_id'],x['trial'])];assert is_strict_success(m)==good(d) and m.false_finish==d['false_finish']; authoritative.append(is_strict_success(m));per_task[x['task_id']]+=int(is_strict_success(m))
                n=sum(authoritative);assert len(raw)==summary['sample_count'] and n==sum(good(d) for d in summary['details'])
                fresh_metrics[a][group]={'runs':len(raw),'strict_successes':n,'false_finish_count':sum(d['false_finish'] for d in summary['details']),'pass3_tasks':sum(c==3 for c in per_task.values()),'tasks':len(per_task),'rescore_drift':0,'trace_sha256':sha(OLD/label/'traces.jsonl'),'job_gpu':read(p)['gpu_index']}
            else:
                remaining.append(label)
                if p.exists():
                    job=read(p);assert job['state']=='running',job
                    running.append(label)
    completed_new=360+sum(read(OLD/p/'summary.json')['sample_count'] for p in done);remaining_runs=822-completed_new
    resource=[]
    for index in [4]:
        raw=subprocess.check_output(['nvidia-smi',f'--id={index}','--query-gpu=index,uuid,name,memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
        idx,uuid,name,memory,free=[v.strip() for v in raw]
        apps=subprocess.check_output(['nvidia-smi',f'--id={index}','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines()
        resource.append({'gpu_index':int(idx),'name':name,'used_mib':int(memory),'free_mib':int(free),'active_compute_processes':sum(s.split(',')[0].strip()==uuid for s in apps)})
    write(R/'training-and-development-audit.json',{'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'all_registered_inputs_unchanged':True,'training':training,'new_optimizer_steps_completed':444,'training_row_exposures_completed':1772,'all_four_final_models_frozen_before_inference':True,'completed_paired_development_comparisons':24,'authoritative_rescore_drift':0,'metrics':metrics,'previous_strict_totals':totals,'real_minus_controls_on_previous_requests':effects,'fresh_source_iid_only':{'runs':54,'strict_successes':sum(source_fresh),'authoritative_rescore_drift':0,'complete_fresh66_claim':False},'completed_new_runs':completed_new,'reused_development_runs':270,'remaining_new_runs':remaining_runs,'remaining_job_labels':remaining,'running_job_labels':running,'completed_fresh_job_labels':done,'fresh_metrics':fresh_metrics,'all_confirmations_complete':remaining_runs==0,'new_algorithm_gain_claim':False,'resource_snapshot':resource,'scope':'OldDev and observedRound012requests are development. Only exit0 complete fresh groups rescored; in-flight groups excluded. Final conclusion requires allgroups and originalfinalizer.'})
    write(R/'completed-development-behavior.json',{'changed_tasks':changed,'strict_counts':counts,'training_tuning_from_these_outputs':False})
    print(json.dumps({'steps':444,'previous_totals':totals,'remaining_new_runs':remaining_runs,'development_rescore_drift':0},ensure_ascii=False))
if __name__=='__main__':main()
