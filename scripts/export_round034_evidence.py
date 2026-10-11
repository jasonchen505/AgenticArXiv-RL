import os,sys,json,datetime,re,ast,tarfile,subprocess,shutil
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,'scripts')
from round034_core import R,D,read,sha,write
def main():
 assert not (R/'experiment-at-report.json').exists();src=R/'source-copies';src.mkdir(exist_ok=True)
 for name in ['round034_core.py','audit_round034_closure.py','prepare_round034_supervision.py','train_round034_plan.py','run_round034_eval.py','finalize_round034_study.py','run_round034_study.py','register_round034_study.py','export_round034_evidence.py','run_round034_preflight.py','check_round034_longest_batch.py','launch_round034_longest.py']:shutil.copyfile(Path('scripts')/name,src/name)
 old=R.parent/'round-033-2026-10-10';closed=R/'closed-R033';closed.mkdir()
 for name in ['pilot-plan.json','pilot-outcome.json','pilot-status.json','pilot-controller.log','planner-predictions.jsonl','planner-predictions-before-execution.json','planner-job.json','reactA-job.json','reactB-job.json']:shutil.copyfile(old/name,closed/name)
 for name in ['planner','reactA','reactB']:shutil.copytree(old/name,closed/name)
 data=R/'TRAIN-supervision';data.mkdir()
 for name in ['plan-supervision.jsonl','mixed-supervision.jsonl']:shutil.copyfile(D/name,data/name)
 jobs={};logs={};learning={};p=read(R/'experiment-plan.json');launch=read(R/'launch.json')
 for label in [a+'-train' for a in p['learning_arms']]+list(p['eval_jobs']):
  f=R/(label+'-job.json')
  if f.exists():jobs[label]=read(f);write(R/('snapshot-'+f.name),jobs[label])
  f=R/(label+'.log')
  if f.exists():text=f.read_text();(R/('snapshot-'+f.name)).write_text(text);logs[label]={'log_SHA256':sha(R/('snapshot-'+f.name)),'task_starts':len(re.findall(r'^\[\d+/\d+\] task=',text,re.M)),'plan_predictions':len(re.findall(r'^Plan prediction ',text,re.M))}
  if label.endswith('-train'):
   records=[]
   if f.exists():
    for line in f.read_text().replace('\r','\n').splitlines():
     if line.startswith("{'loss':"):
      row=ast.literal_eval(line);records.append(row)
   arm=label[:-6];final=Path(p['model_paths'][arm]);m=final/'training_manifest.json';learning[arm]={'logged_optimizer_steps':len(records),'latest_logged':records[-1] if records else None,'final_manifest_present':m.exists()}
   if m.exists():shutil.copyfile(m,R/('snapshot-'+arm+'-training-manifest.json'));learning[arm]['completed_manifest']=read(m)
 state=read(R/'experiment-status.json') if (R/'experiment-status.json').exists() else {'phase':'initializing'};alive=subprocess.run(['kill','-0',str(launch['pid'])],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
 write(R/'experiment-at-report.json',{'captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'launch':launch,'status':state,'controller_alive':alive,'jobs':jobs,'logs':logs,'learning':learning,'closed_training_jobs':sum(jobs.get(a+'-train',{}).get('exit_code')==0 for a in p['learning_arms']),'closed_inference_jobs':sum(jobs.get(k,{}).get('exit_code')==0 for k in p['eval_jobs']),'planned_optimizer_steps_each_arm':207,'full_candidate_efficacy_not_yet_confirmed':True,'source_SFT_remains_default':True})
 exclude={'experiment-status.json','controller.log','experiment.lock'}|{a+'-train'+x for a in p['learning_arms'] for x in ['-job.json','.log']}|{a+x for a in p['eval_jobs'] for x in ['-job.json','.log']};files=[f for f in R.rglob('*') if f.is_file() and (f.parent!=R or f.name not in exclude)];archive=R.parent/'round034-evidence.tar.gz';assert not archive.exists()
 with tarfile.open(archive,'w:gz') as tar:
  for f in sorted(files):tar.add(f,arcname=str(f.relative_to(R.parent)),recursive=False)
 print(json.dumps({'static_files':len(files),'archive_sha256':sha(archive),'snapshot':read(R/'experiment-at-report.json')},ensure_ascii=False,default=str))
if __name__=='__main__':main()
