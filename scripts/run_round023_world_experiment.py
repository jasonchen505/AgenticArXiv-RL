#!/usr/bin/env python3
"""Finite four-arm matched learning, full final freeze, then confirmation."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,shutil,subprocess,time
R=Path('artifacts/reproduction/round-023-2026-10-07');PY='.venv-repro/bin/python';read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def completed_steps():return sum(read(p)['actual_steps'] for p in Path('outputs/repro_round023').glob('*/final/training_manifest.json'))
def state(phase,**kw):write(R/'experiment-status.json',dict(phase=phase,updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed_training_steps=completed_steps(),**kw))
def verify(plan):
    for p,h in plan['sha256'].items():assert sha(p)==h,p
def admission(training,gpu):
    assert gpu in [4,7]
    raw=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=uuid,name,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',');uuid,name,free=[x.strip() for x in raw]
    rows=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True);active=sum(row.split(',')[0].strip()==uuid for row in rows.splitlines())
    return name=='NVIDIA GeForce RTX 4090' and int(free)>=(23000 if training else 8192) and (not training or active==0),{'free_mib':int(free),'compute_processes':active,'uuid':uuid}
def wait_device(training,label):
    deadline=time.monotonic()+(5400 if training else 1800)
    while True:
        devices={}
        for gpu in ([7,4] if training else [4,7]):
            ok,info=admission(training,gpu);devices[str(gpu)]=info
            if ok:return gpu
        state('waiting_for_training_capacity' if training else 'waiting_for_inference_capacity',next_job=label,devices=devices,allowed_physical_gpus=[4,7])
        assert time.monotonic()<deadline,'Finite authorized GPU wait expired';time.sleep(30)
def main():
    with (R/'experiment.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=read(R/'experiment-plan.json');verify(plan);assert plan['allowed_physical_gpus']==[4,7]
        for name in plan['learning_arms']:
            record=R/(name+'-train-job.json')
            if record.exists():assert read(record).get('exit_code')==0;continue
            arm=name[:-2];seed=name[-2:];gpu=wait_device(True,name);verify(plan);state('training',arm=name,physical_gpu=gpu)
            cmd=[PY,'-u','scripts/train_round023_worlds.py','--arm',arm,'--seed',seed,'--output','outputs/repro_round023/'+name]
            subprocess.run([PY,'scripts/run_round020_gpu_job.py','--record',str(record),'--log',str(R/(name+'-train.log')),'--gpu',str(gpu),'--']+cmd,timeout=5400,check=True)
        verify(plan)
        import torch
        from safetensors.torch import load_file
        src=load_file('outputs/repro_20261003/sft_e1/final/adapter_model.safetensors');models={}
        for name in plan['learning_arms']:
            p=Path('outputs/repro_round023')/name/'final';m=load_file(str(p/'adapter_model.safetensors'));ref=load_file(str(p/'ref/adapter_model.safetensors'));training=read(p/'training_manifest.json')
            assert set(src)==set(m)==set(ref) and len(src)==392
            assert all(torch.equal(src[k],ref[k]) for k in src) and sum(not torch.equal(src[k],m[k]) for k in src)==392
            assert training['actual_steps']==310 and training['dataset_rows']==1239 and training['dpo_loss_type']==['sft']
            models[name]={'model':str(p),'changed_adapter_tensors':392,'reference_equal_source_tensors':392,'sha256':{str(f):sha(f) for f in p.rglob('*') if f.is_file()}}
        frozen=R/'all-candidates-frozen-before-inference.json'
        if not frozen.exists():write(frozen,{'recorded_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actual_training_steps':1240,'row_exposures':4956,'models':models})
        else:
            for m in read(frozen)['models'].values():
                for p,h in m['sha256'].items():assert sha(p)==h,p
        for group,g in plan['groups'].items():
            for model in ['sft']+plan['learning_arms']:
                if group=='dev' and model=='sft':continue
                label=group+'-'+model;record=R/(label+'-job.json')
                if record.exists():assert read(record).get('exit_code')==0;continue
                gpu=wait_device(False,label);verify(plan)
                for m in models.values():
                    for p,h in m['sha256'].items():assert sha(p)==h,p
                scratch=Path('data/repro_round023/benchmark_scratch')
                if scratch.exists():shutil.rmtree(scratch)
                env=dict(os.environ)
                for key,name in [('PDF_RAW_PATH','raw'),('PDF_TRANSLATED_PATH','translated'),('PDF_FIGURES_PATH','figures')]:env[key]=str((scratch/name).resolve())
                path='outputs/repro_20261003/sft_e1/final' if model=='sft' else models[model]['model']
                cmd=[PY,'-u','scripts/run_frozen_task_file.py','--tasks',g['tasks'],'--tasks_sha256',plan['sha256'][g['tasks']],'--split',g['split'],'--snapshot',g['snapshot'],'--model',path,'--output',str(R/label),'--prefix','bench_r20261003_075529','--seed','45','--repeat','3']
                state('evaluating',current_job=label,physical_gpu=gpu)
                subprocess.run([PY,'scripts/run_round020_gpu_job.py','--record',str(record),'--log',str(R/(label+'.log')),'--gpu',str(gpu),'--']+cmd,timeout=3600,check=True,env=env)
        state('CPU_finalization');verify(plan);subprocess.run([PY,'scripts/finalize_round023_worlds.py'],timeout=1200,check=True)
if __name__=='__main__':
    try:main()
    except BlockingIOError:raise SystemExit('Existing world experiment controller owns lock')
    except Exception as e:state('failed_or_capacity_timeout',error=str(e));raise
