#!/usr/bin/env python3
"""Freeze 2x2 data/retention study; historical CE arms reused, prospective programs untouched."""
from pathlib import Path
import ast,collections,datetime,hashlib,importlib.metadata,json,os,random
R=Path('artifacts/reproduction/round-027-2026-10-08');OLD=R.parent/'round-024-2026-10-07';PREP=R.parent/'round-026-2026-10-08'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def literal_config(path):
 tree=ast.parse(Path(path).read_text());common={};direct={}
 for node in ast.walk(tree):
  if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='dict' and any(k.arg=='num_train_epochs' for k in node.keywords):
   for k in node.keywords:
    try:common[k.arg]=ast.literal_eval(k.value)
    except (ValueError,TypeError):pass
  if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='DPOConfig':
   for k in node.keywords:
    if k.arg is not None:
     try:direct[k.arg]=ast.literal_eval(k.value)
     except (ValueError,TypeError):pass
 return dict(common,**direct)
def main():
 assert not (R/'experiment-plan.json').exists() and not Path('outputs/repro_round027').exists();os.environ['CUDA_VISIBLE_DEVICES']=''
 pre=read(R/'preflight-result.json');job=read(R/'preflight2-job.json');audit=read(PREP/'retention-independent-audit.json');assert pre['passed'] and job['exit_code']==0 and job['gpu_index']==7 and pre['optimizer_steps']==0 and audit['passed'] and not audit['learning_permitted'];assert job['sampled_max_device_used_mib']<23000
 for p,h in pre['sha256'].items():assert sha(p)==h,p
 old=read(OLD/'experiment-plan.json');files=old['sha256'].copy()
 for p,h in files.items():assert sha(p)==h,p
 for p,h in audit['files_sha256'].items():assert sha(p)==h,p;files[p]=h
 # Independent recipe comparison through literal source configuration, not mirrored field assertions.
 keys=['num_train_epochs','per_device_train_batch_size','gradient_accumulation_steps','learning_rate','max_length','bf16','fp16','logging_steps','save_strategy','optim','gradient_checkpointing','gradient_checkpointing_kwargs','lr_scheduler_type','warmup_ratio','weight_decay','beta','precompute_ref_log_probs','disable_dropout']
 c0=literal_config('scripts/train_round024_curriculum.py');c1=literal_config('scripts/train_round027_retention.py');assert all(c0[k]==c1[k] for k in keys),{k:(c0.get(k),c1.get(k)) for k in keys if c0.get(k)!=c1.get(k)}
 historical=read(OLD/'all-candidates-frozen-before-inference.json');models={'sft':old['source_model']};reused={g:{} for g in ['olddev','newdev']}
 for g in reused:reused[g]['sft']=old['reused_source'][g]
 for arm in old['learning_arms']:
  label='plain'+arm;models[label]=historical['models'][arm]['model']
  for p,h in historical['models'][arm]['sha256'].items():assert sha(p)==h,p;files[p]=h
  for g in reused:reused[g][label]=str(OLD/(g+'-'+arm))
 learning=['klrepeat43','klrepeat44','klcurriculum43','klcurriculum44']
 models.update({name:'outputs/repro_round027/'+name+'/final' for name in learning})
 groups={g:old['groups'][g].copy() for g in ['olddev','newdev']}
 for g,world in [('freshA','trainA'),('freshB','trainB')]:
  label='confirm'+g[-1];splitfile=R/(g+'-splits.json');write(splitfile,{'split':{'iid_test':[t['id'] for t in read(PREP/(label+'-oracles.json'))]}})
  groups[g]={'tasks':str(PREP/(label+'-oracles.json')),'oracles':str(PREP/(label+'-oracles.json')),'split':str(splitfile)+':iid_test','snapshot':'data/repro_round024/'+world+'-snapshot.json'}
 paths=[Path(__file__)]+[R/n for n in ['preflight-result.json','preflight-plan-v2.json','preflight2-job.json']]+[PREP/n for n in ['retention-independent-audit.json','retention-preparation-plan.json','retention-preparation-result.json','train-terminal-exposure-audit.json']]
 for n in ['round027_retention_core.py','train_round027_retention.py','preflight_round027_retention.py','run_round027_retention.py','finalize_round027_retention.py','audit_round027_status.py','run_round020_gpu_job.py','run_frozen_task_file.py','compare_repro_evals.py']:paths.append(Path('scripts')/n)
 for g in groups.values():paths += [Path(g[k]) for k in ['tasks','oracles','snapshot']]+[Path(g['split'].rsplit(':',1)[0])]
 for group,models_reused in reused.items():
  for d in models_reused.values():paths += [Path(d)/n for n in ['traces.jsonl','summary.json','execution_protocol.json']]
 for p in paths:files[str(p)]=sha(p)
 for p in [Path('.venv-repro/lib/python3.10/site-packages/trl/trainer/dpo_trainer.py'),Path('.venv-repro/lib/python3.10/site-packages/trl/trainer/utils.py'),Path('.venv-repro/lib/python3.10/site-packages/transformers/trainer.py')]:files[str(p)]=sha(p)
 orders={}
 for seed in [43,44]:
  order=list(range(443));random.Random(seed).shuffle(order);orders[str(seed)]=[order[i%443] for i in range(795)];assert orders[str(seed)]==pre['anchor_order_seed'+str(seed)]
 rules=['each_seed_each_fresh_world_course_gain_at_least3_vs_KL_repeat','oldDev_strict_at_least_source_and_KL_repeat','oldDev_FF_no_worse_than_source_and_KL_repeat','newDev_strict_retained_vs_plain_course_and_KL_repeat','newDev_FF_no_worse_than_plain_course_and_KL_repeat','fresh_strict_retained_vs_plain_course','fresh_FF_no_worse_than_KL_repeat']
 plan={'registered_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'learning_permitted':True,'question':'Can frozen source-SFT completion-distribution retention preserve old skills while retaining a curriculum benefit on two prospective complete programs?','source_model':old['source_model'],'learning_arms':learning,'model_paths':models,'historical_plain_CE_models_reused':True,'historical_CE_checkpoint_training_before_new_confirmation_generation':True,'seeds':[43,44],'rows_each':795,'steps_each':199,'planned_new_learning_steps':796,'chosen_CE_row_exposures':3180,'anchor_row_exposures':3180,'anchor_rows_unique':443,'anchor_orders':orders,'coefficient':.1,'objective':'Unchanged native DPOTrainer chosen CE plus0.1 mean-over-completion,full-vocabulary forwardKL(sourceSFT||policy), teacher-forced originalTRAIN anchors. No preference/GRPO objective.','native_CE_recipe':{k:c1[k] for k in keys},'gradient_execution':'Native CE super.training_step then anchor backward; same current3/4 accumulation normalization; optimizer/clip only after gradient sum. Default adapter trainable,ref frozen equal originalSFT.','extra_compute_caveat':'Historical CE has same795rows199steps/optimizer but no795extra teacher+policy anchor forwards. KL effect includes extra forward/backward compute;not walltime/token/FLOPmatched. Course/repeat KL arms share anchor stream and budget withinseed.','data_unchanged':'R024795row datasets unchanged including newFINISHsemanticmode curriculum64completed vsrepeat62completed+2blocked. Not puretool-order causalclaim. Anchors all443originalTRAIN chosen;no heldout/gold traces training.','groups':groups,'models_order':list(models),'reused_evals':reused,'trials_per_task':3,'generation_seed':45,'after_baseline_trial_runs':1152,'historical_trial_runs_reused':360,'total_trial_scores':1512,'training_jobs':4,'inference_jobs':26,'jobs_excluding_preflight':30,'comparisons':40,'success_rule':{k:True for k in rules},'numeric_rule_description':'ALL,eachseed: EACHfreshworld KLcourse strict>=KLrepeat+3/48 and>=same-seed plaincourse;oldDevstrict>=max(10/24,KLrepeat);oldDevFF<=min(6/24,KLrepeat);newDevstrict>=max(plaincourse,KLrepeat);newDevFF<=min(plaincourse,KLrepeat);EACHfreshworldFF<=KLrepeat. Close fixed0.1retentionrecipe ifanyfails. No coefficient,seed,case/checkpointselection orautomaticpromotion.','scope':'CFAS/CFSA grammar-selected beforegold/modeloutputs,full signatures excluded alloriginal train/benchmark+allR024splits. Twofamilies*8parametercombos/world,three correlatedtrials/task,tworeusedoldtrainingpaperworlds,sharedtools/params/subsequences. OldR024confirmationoutputs diagnostic-only,no training fromthem. Not paper-blind/broadunseenprogram/significanceclaim.','all4_new_finals_frozen_before_new_candidate_and_fresh_confirmation_inference':True,'no_automatic_promotion':True,'failure_closes_fixed_recipe':True,'allowed_physical_gpus':[4,7],'training_admission':{'minimum_free_mib':23000,'maximum_existing_UUID_compute_processes':0},'inference_admission':{'minimum_free_mib':8192,'shared_allowed':True},'finite_controller_timeout_seconds':43200,'versions':{n:importlib.metadata.version(n) for n in ['torch','transformers','trl','peft','bitsandbytes','datasets']},'sha256':files}
 write(R/'integration-independent-audit.json',{'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'recipe_literal_AST_equal_keys':keys,'native_recipe':plan['native_CE_recipe'],'no_optimizer_step_before_registration':True,'zero_step_preflight_passed':True,'memory_sampled_peak_mib':job['sampled_max_device_used_mib'],'old_scientific_inputs_and_models_unchanged':True,'all443TRAINanchors_and32gold_verified':True,'new_confirmation_model_predictions':0,'new_training_steps':0})
 plan['sha256'][str(R/'integration-independent-audit.json')]=sha(R/'integration-independent-audit.json');write(R/'experiment-plan.json',plan)
 print(json.dumps({k:plan[k] for k in ['registered_at_utc','planned_new_learning_steps','after_baseline_trial_runs','total_trial_scores','comparisons']},indent=2))
if __name__=='__main__':main()
