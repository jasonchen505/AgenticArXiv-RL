"""Snapshot live progress under new names; export closed evidence without mutation."""
import os,sys,json,datetime,tarfile,shutil,subprocess,re
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,'scripts')
from round031_study_core import read,write,sha
R=Path('artifacts/reproduction/round-033-2026-10-10');OLD=R.parent/'round-032-2026-10-10'
def main():
 assert not (R/'pilot-at-report.json').exists();src=R/'source-copies';src.mkdir(exist_ok=True)
 for name in ['audit_round033_parity.py','round033_plan_agent.py','prepare_round033_plan.py','run_round033_pilot.py','run_round033_controller.py','register_round033_pilot.py','finalize_round033_pilot.py','export_round033_evidence.py']:shutil.copyfile(Path('scripts')/name,src/name)
 closed=R/'closed-R032';closed.mkdir();names=['parity-plan.json','parity-outcome.json','parity-status.json','parity-launch.json','parity-controller.log','parity-native-job.json','parity-budget-job.json','parity-native.log','parity-budget.log']
 for name in names:shutil.copyfile(OLD/name,closed/name)
 for mode in ['native','budget']:shutil.copytree(OLD/('parity-'+mode),closed/('parity-'+mode))
 for name in ['plan-CPU.log','parity-audit.log']:shutil.copyfile(Path('data/repro_round033')/name,R/name)
 launch=read(R/'pilot-launch.json');status=read(R/'pilot-status.json') if (R/'pilot-status.json').exists() else {'phase':'initializing'};alive=subprocess.run(['kill','-0',str(launch['pid'])],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0;jobs={};logs={};closed_jobs=0
 for job in ['planner','reactA','reactB']:
  f=R/(job+'-job.json')
  if f.exists():jobs[job]=read(f);write(R/('snapshot-'+f.name),jobs[job]);closed_jobs+=jobs[job].get('exit_code')==0
  f=R/(job+'.log')
  if f.exists():text=f.read_text();(R/('snapshot-'+f.name)).write_text(text);logs[job]={'completed_plan_predictions':len(re.findall(r'^Plan prediction \d+ /48 ',text,re.M)),'ReAct_trial_starts':len(re.findall(r'^\[\d+/48\] task=',text,re.M))}
 pf=R/'planner-predictions-before-execution.json'
 if pf.exists():assert read(pf)['sha256']==sha(R/'planner-predictions.jsonl')
 write(R/'pilot-at-report.json',{'captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'launch':launch,'status':status,'controller_alive':alive,'jobs':jobs,'logs':logs,'closed_jobs':closed_jobs,'registered_jobs':3,'full_outcome_present':(R/'pilot-outcome.json').exists(),'TRAIN_feasibility_only':True,'new_learning_steps':0,'no_partial_gain_claim':True})
 files=[f for f in R.rglob('*') if f.is_file() and f.name not in ['pilot-status.json','pilot-controller.log','pilot.lock','planner-job.json','reactA-job.json','reactB-job.json','planner.log','reactA.log','reactB.log']];archive=R.parent/'round033-evidence.tar.gz';assert not archive.exists()
 with tarfile.open(archive,'w:gz') as tar:
  for f in sorted(files):tar.add(f,arcname=str(f.relative_to(R.parent)),recursive=False)
 print(json.dumps({'archive':str(archive),'sha256':sha(archive),'static_files':len(files),'snapshot':read(R/'pilot-at-report.json')},ensure_ascii=False))
if __name__=='__main__':main()
