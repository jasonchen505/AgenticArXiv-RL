#!/usr/bin/env python3
"""Finite two-seed CE observation ablation; idle GPU4/GPU7 training, frozen diagnostics."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,shutil,subprocess,time
R=Path('artifacts/reproduction/round-019-2026-10-06');OLD=R.parent/'round-014-2026-10-05';WORLD=R.parent/'round-017-2026-10-06';PY='.venv-repro/bin/python'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def state(phase,**kw):write(R/'experiment-status.json',{'phase':phase,'updated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw})
def verify(plan):
    for p,h in plan['sha256'].items():assert sha(p)==h,p
def admission(training,gpu):
    assert gpu in [4,7]
    row=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=uuid,name,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
    uuid,name,free=[x.strip() for x in row]
    raw=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    active=sum(x.split(',')[0].strip()==uuid for x in raw.splitlines())
    return name=='NVIDIA GeForce RTX 4090' and int(free)>=(23000 if training else 8192) and (not training or active==0),int(free),active
def wait_device(training,label):
    deadline=time.monotonic()+(5400 if training else 1800)
    while True:
        devices={}
        for gpu in ([7,4] if training else [4,7]):
            ok,free,active=admission(training,gpu);devices[str(gpu)]={'free_mib':free,'compute_processes':active}
            if ok:return gpu
        state('waiting_for_authorized_training_capacity' if training else 'waiting_for_authorized_inference_capacity',next_job=label,devices=devices,allowed_physical_gpus=[4,7],minimum_free_mib=23000 if training else 8192,new_training_steps_completed=sum(read(p)['actual_steps'] for p in Path('outputs/repro_round019').glob('*/final/training_manifest.json')))
        if time.monotonic()>deadline:raise TimeoutError('Finite GPU4/GPU7 capacity wait expired; do not use other GPU or preempt.')
        time.sleep(30)
def main():
    with (R/'experiment.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=read(R/'experiment-plan.json');verify(plan)
        amendment=read(R.parent/'round-020-2026-10-06'/'operational-amendment.json');assert sha(R/'experiment-plan.json')==amendment['original_plan_sha256'];assert amendment['allowed_physical_gpus']==[4,7]
        plan['sha256'].update(amendment['new_source_sha256']);verify(plan)
        for seed in [43,44]:
            arm='obs'+str(seed);record=R/(arm+'-train-job.json')
            if record.exists():assert read(record).get('exit_code')==0;continue
            gpu=wait_device(True,arm+'-train');verify(plan);state('training',arm=arm,physical_gpu=gpu,allowed_physical_gpus=[4,7])
            cmd=[PY,'-u','scripts/train_round020_observations.py','--objective','sft','--seed',str(seed),'--output','outputs/repro_round019/'+arm]
            subprocess.run([PY,'scripts/run_round020_gpu_job.py','--record',str(record),'--log',str(R/(arm+'-train.log')),'--gpu',str(gpu),'--']+cmd,check=True)
        verify(plan)
        import sys
        sys.path.insert(0,'AgenticArxiv')
        import torch
        from safetensors.torch import load_file
        src=load_file('outputs/repro_20261003/sft_e1/final/adapter_model.safetensors');freezes={}
        for seed in [43,44]:
            arm='obs'+str(seed);directory=Path('outputs/repro_round019')/arm/'final'
            m=load_file(str(directory/'adapter_model.safetensors'));ref=load_file(str(directory/'ref/adapter_model.safetensors'))
            assert len(src)==len(m)==len(ref)==392 and set(src)==set(m)==set(ref)
            assert all(torch.equal(src[k],ref[k]) for k in src)
            assert sum(not torch.equal(src[k],m[k]) for k in src)==392
            manifest=read(directory/'training_manifest.json');assert manifest['actual_steps']==111 and manifest['dataset_rows']==443 and manifest['dpo_loss_type']==['sft']
            freezes[arm]={'model':str(directory),'sha256':{str(p):sha(p) for p in directory.iterdir() if p.is_file()},'changed_tensors':392,'reference_equal_source_tensors':392}
        freeze=R/'all-candidates-frozen-before-inference.json'
        if not freeze.exists():write(freeze,{'recorded_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'training_steps':222,'models':freezes})
        for group,g in plan['groups'].items():
            for seed in [43,44]:
                arm='obs'+str(seed);label=group+'-'+arm;record=R/(label+'-job.json');output=R/label
                if record.exists():assert read(record).get('exit_code')==0;continue
                gpu=wait_device(False,label);verify(plan)
                for f in freezes.values():
                    for p,h in f['sha256'].items():assert sha(p)==h,p
                scratch=Path('data/repro_round019/benchmark_scratch')
                if scratch.exists():shutil.rmtree(scratch)
                env=dict(os.environ)
                for key,name in [('PDF_RAW_PATH','raw'),('PDF_TRANSLATED_PATH','translated'),('PDF_FIGURES_PATH','figures')]:env[key]=str((scratch/name).resolve())
                cmd=[PY,'-u','scripts/run_frozen_task_file.py','--tasks',g['tasks'],'--tasks_sha256',plan['sha256'][g['tasks']],'--split',g['split'],'--model',freezes[arm]['model'],'--snapshot',g['snapshot'],'--output',str(output),'--prefix','bench_r20261003_075529','--seed','45','--repeat','3']
                state('evaluating',current_job=label,physical_gpu=gpu,allowed_physical_gpus=[4,7])
                subprocess.run([PY,'scripts/run_round020_gpu_job.py','--record',str(record),'--log',str(R/(label+'.log')),'--gpu',str(gpu),'--']+cmd,env=env,check=True)
        metrics={};comparisons=[]
        for group,g in plan['groups'].items():
            for seed in [43,44]:
                arm='obs'+str(seed)
                for control in ['sft','replay'+str(seed)]:
                    target=R/(control+'-vs-'+arm+'-'+group+'.json')
                    if not target.exists():subprocess.run([PY,'scripts/compare_repro_evals.py','--base',g['controls'][control],'--sft',str(R/(group+'-'+arm)),'--oracles',g['oracles'],'--output',str(target),'--base_label',control,'--sft_label',arm,'--training_description','Observation-aligned full443row chosen CE versus frozen original CE,2seeds,111steps; already observed diagnostic cases,not blind confirmation','--evaluation_split',group],check=True)
                    values=read(target)['models'];assert all(v['authoritative_rescore_drift']==0 for v in values.values());metrics.setdefault(group,{}).update(values);comparisons.append(str(target))
        n=sum(read(R/(g+'-obs'+str(s))/'summary.json')['sample_count'] for g in plan['groups'] for s in [43,44]);assert n==312
        verify(plan)
        gains={world:{str(seed):round(metrics[world+'-iid']['obs'+str(seed)]['runs']*metrics[world+'-iid']['obs'+str(seed)]['strict_success_rate'])+round(metrics[world+'-ood']['obs'+str(seed)]['runs']*metrics[world+'-ood']['obs'+str(seed)]['strict_success_rate'])-sum(round(metrics[world+'-'+part]['replay'+str(seed)]['runs']*metrics[world+'-'+part]['replay'+str(seed)]['strict_success_rate']) for part in ['iid','ood']) for seed in [43,44]} for world in ['known','transfer']}
        dev={str(seed):round(metrics['dev']['obs'+str(seed)]['runs']*metrics['dev']['obs'+str(seed)]['strict_success_rate']) for seed in [43,44]}
        ff_retained=all(metrics[group]['obs'+str(seed)]['false_finish_rate']<=metrics[group]['replay'+str(seed)]['false_finish_rate'] for group in ['known-iid','transfer-iid'] for seed in [43,44])
        success=all(v>0 for world in gains.values() for v in world.values()) and dev['43']>=11 and dev['44']>=10 and ff_retained
        write(R/'observation-outcome.json',{'new_runs':312,'training_steps':222,'metrics':metrics,'comparisons':comparisons,'CE_matched_effects':gains,'Dev_counts':dev,'false_finish_retained':ff_retained,'registered_success_rule_met':success,'no_automatic_promotion':True,'scope':'Post-hoc diagnostic of all-train observation alignment; cases/worlds already observed,not blind generalization confirmation.'})
        state('completed',new_runs=312,training_steps=222)
if __name__=='__main__':
    try:main()
    except BlockingIOError:raise SystemExit('Controller already owns lock.')
    except Exception as exc:state('failed_or_capacity_timeout',error=str(exc));raise

