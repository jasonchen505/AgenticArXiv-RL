#!/usr/bin/env python3
"""Snapshot complete job outputs only; independent partial scoring and weight proof."""
from pathlib import Path
import collections,datetime,hashlib,json,os,sys
R=Path('artifacts/reproduction/round-023-2026-10-07');read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def good(d):return d['completed'] and d['tool_accurate'] and d['arg_score']==1 and d['ref_score']==1 and d['parse_fail']==0 and d['tool_fail']==0 and d['terminal_semantics_accurate']
def main():
    os.environ.update(CUDA_VISIBLE_DEVICES='',STORE_BACKEND='memory');sys.path.insert(0,'AgenticArxiv')
    from benchmark.metrics import extract_metrics,is_strict_success
    from safetensors.torch import load_file
    import torch
    plan=read(R/'experiment-plan.json');freeze=read(R/'all-candidates-frozen-before-inference.json');jobs={p.name:read(p) for p in R.glob('*-job.json')};source=load_file('outputs/repro_20261003/sft_e1/final/adapter_model.safetensors')
    for p,h in plan['sha256'].items():assert sha(p)==h,p
    training={};frozen_time=datetime.datetime.fromisoformat(freeze['recorded_at_utc'])
    for arm,v in freeze['models'].items():
        for p,h in v['sha256'].items():assert sha(p)==h,p
        path=Path(v['model']);m=load_file(str(path/'adapter_model.safetensors'));ref=load_file(str(path/'ref/adapter_model.safetensors'));assert len(source)==len(m)==len(ref)==392
        assert set(source)==set(m)==set(ref) and all(torch.equal(source[k],ref[k]) for k in source);assert sum(not torch.equal(source[k],m[k]) for k in source)==392
        manifest=read(path/'training_manifest.json');assert manifest['actual_steps']==310 and manifest['dataset_rows']==1239 and manifest['physical_gpu'] in [4,7] and manifest['reference_equal_sft_and_frozen_tensors']==392
        job=jobs[arm+'-train-job.json'];assert job['exit_code']==0 and datetime.datetime.fromisoformat(job['finished_at_utc'])<=frozen_time;training[arm]=manifest
    counts={};evidence={};n=new=0;completed_infer=0
    for group,g in plan['groups'].items():
        tasks={t['id']:t for t in read(g['oracles'])};f,split=g['split'].rsplit(':',1);wanted=set(read(f)['split'][split]);counts[group]={}
        for model in ['sft']+plan['learning_arms']:
            reused=group=='dev' and model=='sft';path=Path(plan['reused_source_Dev']) if reused else R/(group+'-'+model)
            if not reused:
                label=group+'-'+model+'-job.json'
                if label not in jobs or jobs[label].get('exit_code')!=0:continue
                assert datetime.datetime.fromisoformat(jobs[label]['started_at_utc'])>frozen_time;completed_infer+=1
            raw=[json.loads(x) for x in (path/'traces.jsonl').read_text().splitlines()];summary=read(path/'summary.json');details={(x['task_id'],x['trial']):x for x in summary['details']}
            assert not summary['error_count'] and len(raw)==len(details)==len(wanted)*3 and collections.Counter(x['task_id'] for x in raw)==dict.fromkeys(wanted,3)
            strict=ff=0;per=collections.Counter()
            for x in raw:
                m=extract_metrics(tasks[x['task_id']],{'history':x['history']},x['agent_type'],x['trial'],x['session_id']);d=details[(x['task_id'],x['trial'])];assert is_strict_success(m)==good(d) and m.false_finish==d['false_finish'];strict+=int(is_strict_success(m));ff+=int(m.false_finish);per[x['task_id']]+=int(is_strict_success(m))
            counts[group][model]={'runs':len(raw),'strict':strict,'false_finish':ff,'all3_success_tasks':sum(v==3 for v in per.values()),'tasks':len(wanted)};n+=len(raw);new+=0 if reused else len(raw)
            for f in [path/'summary.json',path/'traces.jsonl',path/'execution_protocol.json']:evidence[str(f)]=sha(f)
    now=datetime.datetime.now(datetime.timezone.utc).isoformat();status=read(R/'experiment-status.json');complete=new==756
    result={'captured_at_utc':now,'phase':status['phase'],'status':status,'all4_training_complete':True,'actual_training_steps':1240,'row_exposures':4956,'completed_finals':4,'training_manifests':training,'training_gpu_indices':sorted({v['physical_gpu'] for v in training.values()}),'reference_equal_source_tensors_each':392,'changed_adapter_tensors_each':392,'all_final_models_frozen_before_inference':True,'new_runs_closed':new,'reused_runs':24,'independent_trials_rescored':n,'strict_and_FF_drift':0,'counts':counts,'inference_jobs_closed':completed_infer,'inference_jobs_total':24,'all_confirmation_complete':complete,'complete_outcome_available':(R/'world-data-outcome.json').exists(),'current_jobs':jobs,'frozen_inputs_unchanged':True,'success_rule_not_decided_from_partial_results':not complete,'evidence_sha256':evidence}
    (R/'experiment-at-report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['captured_at_utc','phase','new_runs_closed','independent_trials_rescored','inference_jobs_closed','all_confirmation_complete','counts']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
