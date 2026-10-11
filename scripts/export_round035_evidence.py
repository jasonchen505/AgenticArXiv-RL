import os,sys,json,datetime,subprocess,shutil,tarfile
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,'scripts')
from round034_core import R,read,sha,write
S=R.parent/'round-035-2026-10-10'
def main():
 assert not (S/'execution-at-report.json').exists();p=read(R/'experiment-plan.json');audit=read(S/'independent-progress-audit.json');closed=S/'closed-R034';closed.mkdir()
 for n in ['experiment-plan.json','launch.json','all-finals-before-inference.json','confirmation-tasks.json','olddev-tasks.json','newdev-tasks.json','supervision-CPU-audit.json','mixed49-train-job.json','lr0_49-train-job.json']:shutil.copyfile(R/n,closed/n)
 for arm in ['mixed49','lr0_49']:shutil.copyfile(Path(p['model_paths'][arm])/'training_manifest.json',closed/(arm+'-training-manifest.json'))
 for label in audit['counts']:
  shutil.copyfile(R/(label+'-job.json'),closed/(label+'-job.json'));shutil.copyfile(R/(label+'.log'),closed/(label+'.log'));shutil.copytree(R/label,closed/label)
 jobs={};logs={}
 for label in p['eval_jobs']:
  f=R/(label+'-job.json')
  if f.exists():jobs[label]=read(f);write(S/('snapshot-'+f.name),jobs[label])
  f=R/(label+'.log')
  if f.exists():shutil.copyfile(f,S/('snapshot-'+f.name));logs[label]={'snapshot_SHA256':sha(S/('snapshot-'+f.name))}
 launch=read(S/'operation-launch.json');alive=subprocess.run(['kill','-0',str(launch['pid'])],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
 state=read(S/'operation-status.json') if (S/'operation-status.json').exists() else {'phase':'initializing'};shutil.copyfile(S/'operation-controller.log',S/'snapshot-operation-controller.log')
 write(S/'execution-at-report.json',{'captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'launch':launch,'status':state,'controller_alive':alive,'audited_completed_units':audit['closed_inference_jobs'],'audited_full_scores':audit['full_trials_rescored'],'jobs':jobs,'logs':logs,'original_failed_job_exit1_preserved':True,'new_training_jobs_this_round':0,'source_SFT_remains_default':True})
 for n in ['audit_round035_progress.py','export_round035_evidence.py']:shutil.copyfile(Path('scripts')/n,S/'source-copies'/n)
 for n in ['round033_plan_agent.py','run_round034_eval.py','finalize_round034_study.py']:shutil.copyfile(Path('scripts')/n,S/'source-copies'/('original-'+n))
 exclude={'operation-status.json','operation-controller.log','operation.lock'};files=[f for f in S.rglob('*') if f.is_file() and (f.parent!=S or f.name not in exclude)];archive=S.parent/'round035-evidence.tar.gz';assert not archive.exists()
 with tarfile.open(archive,'w:gz') as tar:
  for f in sorted(files):tar.add(f,arcname=str(f.relative_to(S.parent)),recursive=False)
 print({'static_files':len(files),'archive_SHA256':sha(archive),'snapshot':read(S/'execution-at-report.json')})
if __name__=='__main__':main()
