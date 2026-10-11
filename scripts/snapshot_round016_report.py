#!/usr/bin/env python3
"""CPU-only report snapshot; independent of finite GPU controller."""
from pathlib import Path
import datetime,fcntl,json,subprocess,sys,time
import run_round016_resume as resume
R=resume.R;OLD=resume.core.R
def read(p):
    for attempt in range(5):
        try:return json.loads(Path(p).read_text())
        except (FileNotFoundError,json.JSONDecodeError):
            if attempt==4:raise
            time.sleep(.2)
def main():
    resume.verify()
    subprocess.run([resume.core.PY,'scripts/audit_round016_progress.py'],check=True)
    s=read(R/'resume-status.json');a=read(R/'training-and-development-audit.json')
    label=s.get('current_job') or s.get('next_job')
    if label and (OLD/(label+'-job.json')).exists():j=read(OLD/(label+'-job.json'))
    else:
        paths=[p for p in OLD.glob('fresh-*-job.json') if read(p).get('memory_admission')]
        assert paths
        p=max(paths,key=lambda p:p.stat().st_mtime);label=p.name.removesuffix('-job.json');j=read(p)
    assert j['gpu_index']==4 and j['memory_admission']['minimum_free_mib']==8192
    with (OLD/'controller.lock').open('r') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:held=True
        else:held=False
    assert held or s['phase'] in ['completed','failed_or_wait_timeout']
    v={'recorded_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'label':label,'state':j['state'],'gpu_index':4,'memory_admission':j['memory_admission'],'process_pid':j['pid'],'elapsed_seconds':j['elapsed_seconds'],'device_samples_at_snapshot':j['device_samples'][-3:],'sampled_device_peak_mib':j['sampled_max_device_used_mib'],'device_peak_includes_other_processes':True,'controller_lock_held':held}
    resume.write(R/'job-resource-snapshot.json',v);resume.write(R/'resume-status-at-report.json',s)
    print(json.dumps({'phase':s['phase'],'current_job':label,'completed_new_runs':a['completed_new_runs'],'remaining_new_runs':a['remaining_new_runs'],'fresh_metrics':a['fresh_metrics']}))
if __name__=='__main__':main()
