import os,sys,shutil,tarfile,datetime,subprocess
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,'scripts')
from round036_core import R,D,read,write,sha
OLD=R.parent/'round-034-2026-10-10';OP=R.parent/'round-035-2026-10-10'
def main():
 # A fixed report snapshot. Never overwrite original R034/R035 reports or live pilot files.
 closed=R/'closed-R034';assert not closed.exists();closed.mkdir();p=read(OLD/'experiment-plan.json')
 for n in ['experiment-plan.json','all-finals-before-inference.json','confirmation-tasks.json','olddev-tasks.json','newdev-tasks.json','supervision-CPU-audit.json','mixed49-train-job.json','lr0_49-train-job.json','launch.json']:shutil.copyfile(OLD/n,closed/n)
 for arm in ['mixed49','lr0_49']:shutil.copyfile(Path(p['model_paths'][arm])/'training_manifest.json',closed/(arm+'-training-manifest.json'))
 for label in p['eval_jobs']:
  shutil.copytree(OLD/label,closed/label)
  for suffix in ['-job.json','.log']:shutil.copyfile(OLD/(label+suffix),closed/(label+suffix))
 op=R/'closed-R035';op.mkdir()
 for n in ['study-outcome.json','CPU-recovery-receipt.json','operation-amendment.json','CPU-startup-amendment.json','operation-launch.json','operation-status.json','operation-controller.log']:shutil.copyfile(OP/n,op/n)
 src=R/'source-copies';src.mkdir(exist_ok=True)
 for n in ['audit_round036_closure.py','round036_core.py','prepare_round036_curriculum.py','train_round036_plan.py','register_round036_pilot.py','run_round036_pilot.py','run_round036_eval.py','finalize_round036_pilot.py','export_round036_evidence.py','resume_round036_empty_output.py','round035_literal_error_log.py','run_round027_retention.py','run_round020_gpu_job.py','round033_plan_agent.py','finalize_round035_operational.py','finalize_round034_study.py','run_round034_eval.py']:shutil.copyfile(Path('scripts')/n,src/n)
 for n in ['plan-supervision.jsonl','training-supervision.jsonl']:shutil.copyfile(D/n,R/n)
 # Snapshot only completed manifests; active logs are explicitly named snapshot.
 launch=read(R/'operation-launch.json') if (R/'operation-launch.json').exists() else read(R/'pilot-launch.json');alive=subprocess.run(['kill','-0',str(launch['pid'])],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0;jobs={}
 for f in R.glob('*-job.json'):
  jobs[f.name]=read(f);shutil.copyfile(f,R/('snapshot-'+f.name))
 for f in [R/'pilot-controller.log',R/'operation-controller.log']+[R/(n+'.log') for n in ['preflight','plan49-train','plan49-train-recovered','lr0_49-train']]+[R/(n+'.log') for n in read(R/'pilot-plan.json')['eval_jobs']]:
  if f.exists():shutil.copyfile(f,R/('snapshot-'+f.name))
 manifest={}
 for arm in ['plan49','lr0_49']:
  f=Path('outputs/repro_round036')/arm/'final/training_manifest.json'
  if f.exists():shutil.copyfile(f,R/(arm+'-training-manifest.json'));manifest[arm]=read(f)
 write(R/'execution-at-report.json',{'captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'launch':launch,'controller_alive':alive,'status':read(R/'operation-status.json') if (R/'operation-status.json').exists() else read(R/'pilot-status.json') if (R/'pilot-status.json').exists() else {'phase':'initializing'},'jobs':jobs,'completed_training_arms':list(manifest),'positive_lr_steps_completed':sum(m['positive_lr_learning_steps'] for m in manifest.values()),'source_SFT_remains_default':True,'no_heldout_model_predictions':True})
 exclude={'pilot-status.json','pilot-controller.log','pilot.lock','operation-status.json','operation-controller.log'};files=[f for f in R.rglob('*') if f.is_file() and (f.parent!=R or f.name not in exclude) and not (f.parent==R and f.name.endswith('.log') and not f.name.startswith('snapshot-') and f.name!='curriculum-CPU.log')];archive=R.parent/'round036-evidence.tar.gz';assert not archive.exists()
 with tarfile.open(archive,'w:gz') as tar:
  for f in sorted(files):tar.add(f,arcname=str(f.relative_to(R.parent)),recursive=False)
 print({'archive_SHA256':sha(archive),'static_files':len(files),'snapshot':read(R/'execution-at-report.json')})
if __name__=='__main__':main()
