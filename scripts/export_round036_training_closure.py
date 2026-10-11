import os,sys,shutil,datetime,tarfile,subprocess
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,'scripts')
from round036_core import R,read,write,sha,verify

def main():
 import torch
 from safetensors.torch import load_file
 p=read(R/'pilot-plan.json');verify(p);dest=R/'training-closure';assert not dest.exists();dest.mkdir();path=Path(p['model_paths']['plan49']);m=read(path/'training_manifest.json');job=read(R/'plan49-train-recovered-job.json');assert job['exit_code']==0 and job['gpu_index']==4;assert m['actual_optimizer_steps']==128 and m['positive_lr_learning_steps']==127 and m['microbatch_calls']==512 and m['order']==p['training_order'];assert [x['row_index'] for x in m['exposures']]==p['training_order'];assert all(x['accumulation_group_size']==4 for x in m['exposures']);source=load_file(p['source_model']+'/adapter_model.safetensors');candidate=load_file(str(path/'adapter_model.safetensors'));assert set(candidate)==set(source) and len(source)==392;changed=sum(not torch.equal(candidate[k],source[k]) for k in source);assert changed==392;assert sha(Path(p['source_model'])/'adapter_model.safetensors')==p['route_policy']['source_adapter_SHA256']
 for src,name in [(path/'training_manifest.json','plan49-training-manifest.json'),(R/'plan49-train-recovered-job.json','plan49-train-recovered-job.json'),(R/'plan49-train-recovered.log','plan49-train-recovered.log'),(Path('scripts/export_round036_training_closure.py'),'export_round036_training_closure.py')]:shutil.copyfile(src,dest/name)
 write(dest/'closed-candidate-audit.json',{'passed':True,'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'candidate_GPU_exit_code':0,'physical_gpu':4,'actual_optimizer_calls':128,'actual_positive_lr_steps':127,'microbatch_calls':512,'order_and_full_exposure_record_verified':True,'changed_adapter_tensors_independently_compared':changed,'source_adapter_SHA_unchanged':True,'candidate_adapter_SHA256':sha(path/'adapter_model.safetensors'),'no_model_success_rate_or_generalization_claim':True,'sha256':{str(f):sha(f) for f in dest.iterdir() if f.is_file()}});write(dest/'snapshot.json',{'captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'operation_status':read(R/'operation-status.json'),'completed_candidate_training_only':True,'lr0_and_inference_not_asserted_complete':True,'source_remains_default':True})
 archive=R.parent/'round036-training-closure.tar.gz'
 with tarfile.open(archive,'w:gz') as t:
  for f in dest.iterdir():t.add(f,arcname=str(f.relative_to(R.parent)),recursive=False)
 print({'audit':read(dest/'closed-candidate-audit.json'),'snapshot':read(dest/'snapshot.json'),'archive_SHA256':sha(archive)})
if __name__=='__main__':main()
