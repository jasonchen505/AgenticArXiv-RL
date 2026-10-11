import os,sys,json,datetime,ast,subprocess,shutil,tarfile
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,'scripts')
from round034_core import R,read,sha,write
def main():
 target=R/'progress-update';assert not target.exists();target.mkdir()
 p=read(R/'experiment-plan.json');jobs={};learning={}
 for label in [a+'-train' for a in p['learning_arms']]+list(p['eval_jobs']):
  f=R/(label+'-job.json')
  if f.exists():jobs[label]=read(f);shutil.copyfile(f,target/f.name)
  log=R/(label+'.log');records=[]
  if log.exists():
   shutil.copyfile(log,target/log.name)
   if label.endswith('-train'):
    for line in log.read_text().replace('\r','\n').splitlines():
     if line.startswith("{'loss':"):records.append(ast.literal_eval(line))
  if label.endswith('-train'):
   arm=label[:-6];mf=Path(p['model_paths'][arm])/'training_manifest.json'
   learning[arm]={'logged_optimizer_steps':len(records),'latest_logged':records[-1] if records else None,'final_manifest_present':mf.exists()}
   if mf.exists():shutil.copyfile(mf,target/(arm+'-training-manifest.json'));learning[arm]['manifest_filename']=arm+'-training-manifest.json'
 for name in ['all-finals-before-inference.json','study-outcome.json']:
  if (R/name).exists():shutil.copyfile(R/name,target/name)
 launch=read(R/'launch.json');alive=subprocess.run(['kill','-0',str(launch['pid'])],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
 write(target/'progress.json',{'captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':read(R/'experiment-status.json'),'controller_alive':alive,'learning':learning,'closed_training_jobs':sum(jobs.get(a+'-train',{}).get('exit_code')==0 for a in p['learning_arms']),'closed_inference_jobs':sum(jobs.get(k,{}).get('exit_code')==0 for k in p['eval_jobs']),'jobs':jobs,'source_SFT_remains_default':True})
 archive=R.parent/'round034-progress-update.tar.gz';assert not archive.exists()
 with tarfile.open(archive,'w:gz') as tar:
  for f in sorted(target.rglob('*')):
   if f.is_file():tar.add(f,arcname=str(f.relative_to(R.parent)),recursive=False)
 print(json.dumps({'archive_SHA256':sha(archive),'progress':read(target/'progress.json')},default=str))
if __name__=='__main__':main()
