from pathlib import Path
import os,json,datetime,subprocess,tarfile,sys
os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,'scripts')
from round031_study_core import R,read,write,sha,collect

def main():
 sys.path.insert(0,'AgenticArxiv');from tools.bootstrap import require_all_tools;require_all_tools('Read-only R031 snapshot');s=collect();p=read(R/'experiment-plan.json');status=read(R/'experiment-status.json');launch=read(R/'launch.json');pid=launch['pid'];ps=subprocess.run(['ps','-p',str(pid),'-o','pid,etime,args'],capture_output=True,text=True);jobs={}
 for f in R.glob('*-job.json'):
  j=read(f);assert j['gpu_index'] in [4,7];jobs[f.name]=j
 label=status.get('current_job');log=R/(str(label)+'.log');lines=log.read_text(errors='replace').splitlines() if log.exists() else [];completed_task_starts=[x for x in lines if x.startswith('[') and '] task=' in x];s.update(captured_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),launch=launch,live_status=status,controller_alive=ps.returncode==0,controller_ps=ps.stdout.strip(),jobs_at_snapshot=jobs,current_task_start_count=len(completed_task_starts),current_log_tail=lines[-35:],full_study_pending=not (R/'study-outcome.json').exists(),planned_trial_scores=p['total_trial_scores'],planned_jobs=p['inference_jobs'],model_gain_claimed=False);write(R/'study-at-report.json',s)
 archive=R.parent/'round031-evidence.tar.gz'
 with tarfile.open(archive,'w:gz') as tar:
  for f in R.iterdir():
   if f.is_file() and not f.name.endswith('.tmp') and f.name not in ['experiment.lock','controller.log','experiment-status.json'] and not f.name.endswith('.log') and not f.name.endswith('-job.json'):tar.add(f,arcname=f.relative_to(R.parent))
  for name,j in jobs.items():
   frozen=R/('snapshot-'+name);write(frozen,j);tar.add(frozen,arcname=frozen.relative_to(R.parent));label=name[:-9];lf=R/(label+'.log')
   if lf.exists():snap=R/('snapshot-'+lf.name);snap.write_text(lf.read_text(errors='replace'));tar.add(snap,arcname=snap.relative_to(R.parent))
   if j.get('exit_code')==0:tar.add(R/label,arcname=(R/label).relative_to(R.parent))
 print({'snapshot_UTC':s['captured_at_utc'],'closed_jobs':s['closed_jobs'],'independent_rescores':s['independently_rescored'],'current_job':label,'task_starts':s['current_task_start_count'],'controller_alive':s['controller_alive'],'archive_sha256':sha(archive)})
if __name__=='__main__':main()
