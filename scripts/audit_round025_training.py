#!/usr/bin/env python3
"""Independent completed training audit before corrected freeze guard resumes."""
from pathlib import Path
import datetime,hashlib,json,os
R=Path('artifacts/reproduction/round-024-2026-10-07');A=R.parent/'round-025-2026-10-07';read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 A.mkdir(parents=True,exist_ok=True);assert not (A/'training-independent-audit.json').exists();os.environ['CUDA_VISIBLE_DEVICES']=''
 import torch
 from safetensors.torch import load_file
 plan=read(R/'experiment-plan.json');old=Path('scripts/run_round024_curriculum_experiment.py');new=Path('scripts/run_round025_curriculum_resume.py');assert old.read_text().replace("training['dataset_rows']==1239","training['dataset_rows']==795")==new.read_text()
 for p,h in plan['sha256'].items():assert sha(p)==h,p
 baseline=read(R/'baseline-frozen-for-learning.json')
 for p,h in baseline['sha256'].items():assert sha(p)==h,p
 source=load_file('outputs/repro_20261003/sft_e1/final/adapter_model.safetensors');assert len(source)==392;models={};trainjobs={};manifests={}
 for name in plan['learning_arms']:
  d=Path('outputs/repro_round024')/name/'final';manifest=read(d/'training_manifest.json');j=read(R/(name+'-train-job.json'));assert j['exit_code']==0 and j['gpu_index']==7 and manifest['dataset_rows']==795 and manifest['actual_steps']==199 and manifest['dpo_loss_type']==['sft'] and manifest['dpo_loss_weights']==[1.] and manifest['seed']==int(name[-2:]) and manifest['lr']==5e-6 and manifest['reference_equal_sft_and_frozen_tensors']==392
  adapter=load_file(str(d/'adapter_model.safetensors'));ref=load_file(str(d/'ref/adapter_model.safetensors'));assert set(source)==set(adapter)==set(ref)
  assert all(torch.equal(source[k],ref[k]) for k in source);changed=sum(not torch.equal(source[k],adapter[k]) for k in source);assert changed==392
  assert datetime.datetime.fromisoformat(plan['registered_at_utc'])<datetime.datetime.fromisoformat(j['started_at_utc']);assert datetime.datetime.fromisoformat(baseline['recorded_at_utc'])<datetime.datetime.fromisoformat(j['started_at_utc'])
  models[name]={'model':str(d),'changed_adapter_tensors':changed,'reference_equal_source_tensors':392,'sha256':{str(p):sha(p) for p in d.rglob('*') if p.is_file()}};trainjobs[name]=j;manifests[name]=manifest
 assert not (R/'all-candidates-frozen-before-inference.json').exists();assert not [p for p in R.glob('*-job.json') if p.name!='source-new-dev-job.json' and not p.name.endswith('-train-job.json')]
 result={'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'actual_training_steps':796,'actual_row_exposures':3180,'rows_each':795,'steps_each':199,'all4_training_complete':True,'successful_training_jobs':4,'all_training_physical_gpu':7,'source_reference_equal_tensors_each':392,'changed_adapter_tensors_each':392,'models':models,'training_manifests':manifests,'training_jobs':trainjobs,'original1215inputs_unchanged':True,'original_source_and_new_source_dev48_frozen_unchanged':True,'only_guard_literal_correction':'dataset_rows==1239 -> ==795, matching original scientific protocol and actual data;all other bytes unchanged','no_candidate_or_confirmation_inference_before_audit':True,'no_new_learning_on_resume':True,'plan_sha256':sha(R/'experiment-plan.json'),'original_controller_sha256':sha(old),'corrected_controller_sha256':sha(new)}
 write(A/'training-independent-audit.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['models','training_jobs','training_manifests']},ensure_ascii=False))
if __name__=='__main__':main()
