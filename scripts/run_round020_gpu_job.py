#!/usr/bin/env python3
"""Execute an audited job and sample device usage without importing CUDA in monitor."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--record', type=Path, required=True)
    p.add_argument('--log', type=Path, required=True)
    p.add_argument('--gpu', type=int, choices=[4,7], required=True,
                   help='Use GPU4 orGPU7 only with at least8GiB free; no preemption.')
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not command:
        raise SystemExit('No command')
    for file in (a.record, a.log):
        file.parent.mkdir(parents=True, exist_ok=True)
        if file.exists():
            raise SystemExit(f'Refusing to overwrite existing job evidence: {file}')
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', STORE_BACKEND='memory')
    row = subprocess.check_output(['nvidia-smi', f'--id={a.gpu}', '--query-gpu=uuid,name,memory.used,memory.free', '--format=csv,noheader,nounits'], text=True, timeout=10).strip().split(',')
    uuid, name, used, free = [x.strip() for x in row]
    raw = subprocess.check_output(['nvidia-smi', f'--id={a.gpu}', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True, timeout=10)
    active = sum(line.split(',')[0].strip()==uuid for line in raw.splitlines())
    if name!='NVIDIA GeForce RTX 4090' or int(free)<8192:
        raise SystemExit('Authorized GPU lacks8GiB memory headroom or is incompatible; no job started.')
    started = time.monotonic()
    record = {'command':command,'gpu_index':a.gpu,'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sample_interval_seconds':5,'device_samples':[],'state':'running'}
    record['memory_admission']={'free_mib':int(free),'used_mib':int(used),'existing_compute_processes':active,'minimum_free_mib':8192,'shared_device':bool(active),'authorized_physical_gpu':a.gpu,'allowed_physical_gpus':[4,7]}
    with a.log.open('w') as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        record['pid'] = process.pid
        while True:
            try:
                raw = subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=5)
                used,util = [int(value.strip()) for value in raw.strip().split(',')]
                record['device_samples'].append({'elapsed_seconds':round(time.monotonic()-started,2),'used_mib':used,'utilization_percent':util})
            except (OSError,ValueError,subprocess.SubprocessError) as exc:
                record.setdefault('monitor_errors',[]).append(str(exc))
            code = process.poll()
            record['elapsed_seconds'] = round(time.monotonic()-started,2)
            record['sampled_max_device_used_mib'] = max((x['used_mib'] for x in record['device_samples']),default=0)
            if code is not None:
                record.update(state='completed' if code==0 else 'failed',exit_code=code,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            a.record.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
            if code is not None:
                break
            time.sleep(5)
    print(json.dumps({key:val for key,val in record.items() if key!='device_samples'},ensure_ascii=False,indent=2))
    raise SystemExit(code)


if __name__=='__main__':
    main()
