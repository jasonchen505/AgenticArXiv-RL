#!/usr/bin/env python3
"""Finite operational continuation of frozen Round014; no new learning."""
from pathlib import Path
import datetime, hashlib, json, os, shutil, subprocess, sys, time
import run_round014_experiment as core
R = Path('artifacts/reproduction/round-015-2026-10-05')
ALLOW = [4, 3, 2, 1, 0]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, value): Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
def stamp(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def devices():
    result=[]
    for index in ALLOW:
        row=subprocess.check_output(['nvidia-smi', f'--id={index}', '--query-gpu=uuid,name,memory.used', '--format=csv,noheader,nounits'], text=True, timeout=10).strip().split(',')
        uuid,name,used=[s.strip() for s in row]
        raw=subprocess.check_output(['nvidia-smi', f'--id={index}', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True, timeout=10)
        count=sum(line.split(',')[0].strip()==uuid for line in raw.splitlines())
        result.append(dict(gpu_index=index,name=name,used_mib=int(used),active_compute_processes=count))
    return result
def verify():
    amendment=core.read(R/'operational-amendment.json')
    assert amendment['allowed_gpu_indices']==[0,1,2,3,4]
    assert sha(core.R/'experiment-plan.json')==amendment['original_plan_sha256']
    for p,h in amendment['operational_source_sha256'].items(): assert sha(p)==h,p
    plan=core.read(core.R/'experiment-plan.json')
    for p,h in {**plan['sha256'],**plan['reused_evidence_sha256']}.items(): assert sha(p)==h,p
    for arm in core.ARMS:
        frozen=core.read(core.R/(arm+'-candidate-freeze.json'))
        assert sha(Path(frozen['model'])/'adapter_model.safetensors')==amendment['candidate_sha256'][arm]==frozen['candidate_sha256']
        assert core.read(core.R/(arm+'-train-job.json'))['exit_code']==0
    return amendment
def state(phase, **kw):
    value={'phase':phase,'updated_at_utc':stamp(),**kw}
    write(R/'resume-status.json',value)
    write(core.R/'controller-status.json',value)
def job(label, command, scratch=None):
    record=core.R/(label+'-job.json')
    if record.exists(): assert core.read(record).get('exit_code')==0,label; return
    assert label in amendment['remaining_job_labels'], label
    assert command[2]=='scripts/run_frozen_task_file.py',command
    assert not label.endswith('-train')
    while True:
        verify()
        inventory=devices()
        idle=[d for d in inventory if d['name']=='NVIDIA GeForce RTX 4090' and d['active_compute_processes']==0 and d['used_mib']<256]
        if idle: break
        state('waiting_for_allowed_gpu',next_job=label,allowed_gpu_indices=[0,1,2,3,4],resource_snapshot=inventory,wait_deadline_utc=deadline_utc)
        if time.monotonic()>=deadline: raise TimeoutError('Two-hour finite resource wait expired; completed evidence retained; no GPU preemption.')
        time.sleep(30)
    index=idle[0]['gpu_index']
    if scratch and scratch.exists():
        assert scratch==Path('data/repro_round014/fresh_scratch'),scratch
        shutil.rmtree(scratch)
    state('running',current_job=label,gpu_index=index)
    # The launcher rechecks idle device immediately before creating the child.
    core.run([core.PY,'scripts/run_round015_available_job.py','--record',str(record),'--log',str(core.R/(label+'.log')),'--gpu',str(index),'--']+command,core.env_for(scratch) if scratch else None)
if __name__=='__main__':
    amendment=verify()
    deadline=time.monotonic()+7200
    deadline_utc=(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=2)).isoformat()
    core.job=job
    try:
        core.main()
        verify()
        state('completed',remaining_new_runs=0)
    except BlockingIOError:
        raise SystemExit('Existing Round014 controller owns lock; no duplicate continuation.')
    except Exception as exc:
        state('failed_or_wait_timeout',error=str(exc)); raise
