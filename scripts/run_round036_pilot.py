import os,sys,signal,fcntl,datetime,time,subprocess
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';sys.path[:0]=['AgenticArxiv','scripts']
from round036_core import R,D,PY,read,rows,sha,write,verify
import run_round027_retention as h

def state(phase,**kw):write(R/'pilot-status.json',dict(phase=phase,updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw))
def wait_gpu(training,label):
 deadline=time.monotonic()+(5400 if training else 1800)
 while True:
  devices={}
  for gpu in [4,7]:
   ok,info=h.admission(training,gpu);devices[str(gpu)]=info
   if ok:return gpu
  state('waiting_capacity',current_job=label,training=training,devices=devices);assert time.monotonic()<deadline,'Authorized capacity timeout';time.sleep(30)
def job(p,label,script_args,training,timeout):
 record=R/(label+'-job.json');assert not record.exists();verify(p);gpu=wait_gpu(training,label);env=dict(os.environ,TMPDIR=str((D/'tmp').resolve()),FIGURE_ANALYSIS_BACKEND='extractive')
 for k,n in [('PDF_RAW_PATH','raw'),('PDF_TRANSLATED_PATH','translated'),('PDF_FIGURES_PATH','figures')]:env[k]=str((D/label/n).resolve())
 state('preflight' if label=='preflight' else 'training' if training else 'evaluating',current_job=label,physical_gpu=gpu);h.run_job([PY,'scripts/run_round020_gpu_job.py','--record',str(record),'--log',str(R/(label+'.log')),'--gpu',str(gpu),'--',PY,'-u']+script_args,timeout,env)
def freeze(p):
 import torch
 from safetensors.torch import load_file
 from transformers import AutoTokenizer
 source=load_file(p['source_model']+'/adapter_model.safetensors');tok=AutoTokenizer.from_pretrained(p['source_model'],local_files_only=True);data=rows(p['training_data']);hashes={};models={}
 for name,path in p['model_paths'].items():
  if name!='source':
   a=load_file(path+'/adapter_model.safetensors');m=read(Path(path)/'training_manifest.json');changed=sum(not torch.equal(a[k],source[k]) for k in source);assert set(a)==set(source) and changed==(392 if name=='plan49' else 0);assert m['actual_optimizer_steps']==128 and m['microbatch_calls']==512 and m['order']==p['training_order'];assert m['positive_lr_learning_steps']==(127 if name=='plan49' else 0);models[name]={'changed_adapter_tensors':changed,'optimizer_calls':128,'positive_lr_steps':m['positive_lr_learning_steps']}
  t=AutoTokenizer.from_pretrained(path,local_files_only=True);assert t.get_vocab()==tok.get_vocab() and t.all_special_tokens==tok.all_special_tokens
  for x in data:assert t.apply_chat_template(x['prompt'],add_generation_prompt=True,tokenize=True)==tok.apply_chat_template(x['prompt'],add_generation_prompt=True,tokenize=True)
  for f in Path(path).rglob('*'):
   if f.is_file():hashes[str(f)]=sha(f)
 assert sha(Path(p['route_policy']['native'])/'adapter_model.safetensors')==p['route_policy']['source_adapter_SHA256'];write(R/'all-finals-before-inference.json',{'recorded_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sha256':hashes,'models':models,'all512_prompt_IDs_equal_source_each_model':True,'native_frozen_by_construction_not_learned_retention':True})
def main():
 signal.signal(signal.SIGTERM,h.terminate);signal.signal(signal.SIGINT,h.terminate)
 with (R/'pilot.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p=read(R/'pilot-plan.json');verify(p)
  job(p,'preflight',['scripts/train_round036_plan.py','--arm','plan49','--preflight'],True,900);assert read(R/'preflight-result.json')['passed'] and read(R/'preflight-result.json')['optimizer_steps']==0
  for arm in p['learning_arms']:job(p,arm+'-train',['scripts/train_round036_plan.py','--arm',arm],True,5400)
  verify(p);freeze(p)
  for label in p['eval_jobs']:job(p,label,['scripts/run_round036_eval.py','--label',label],False,3600)
  state('CPU_finalization');subprocess.run([PY,'scripts/finalize_round036_pilot.py'],check=True,timeout=1800);state('completed')
if __name__=='__main__':
 try:main()
 except Exception as e:state('failed',error=str(e));raise
