#!/usr/bin/env python3
"""Two anchored-DPO candidates, five frozen controls, finite GPU4-only pipeline."""
from pathlib import Path
from datetime import datetime,timezone
import fcntl,hashlib,json,os,shutil,subprocess,sys,time
R=Path('artifacts/reproduction/round-014-2026-10-05');OLD=Path('artifacts/reproduction/round-012-2026-10-04');PY='.venv-repro/bin/python'
SOURCE=Path('outputs/repro_20261003/sft_e1/final')
ARMS=['replay43','real43','replay44','real44'];CONTROLS=['sft','ct43','ct44'];MODELS=CONTROLS+ARMS
PAIRS=[(b,'real'+str(s)) for s in [43,44] for b in ['sft','ct'+str(s),'replay'+str(s)]]+ [('sft','replay'+str(s)) for s in [43,44]]
GROUPS=['dev','previous-iid_test','previous-ood_test','fresh-iid_test','fresh-ood_test']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def stamp():return datetime.now(timezone.utc).isoformat()
def state(phase,**x):
    write(R/'controller-status.json',{'phase':phase,'updated_at_utc':stamp(),**x})
    if Path('scripts/build_round014_report.py').exists():subprocess.run([PY,'scripts/build_round014_report.py'],check=True)
def run(cmd,env=None):print('Executing',cmd,flush=True);subprocess.run(cmd,env=env,check=True)
def leaf(a,g):return a+'-dev' if g=='dev' else g.split('-')[0]+'-'+a+'-'+g.split('-',1)[1]
def env_for(base):
    env=os.environ.copy()
    for key,name in [('PDF_RAW_PATH','pdf_raw'),('PDF_TRANSLATED_PATH','pdf_translated'),('PDF_FIGURES_PATH','pdf_figures')]:env[key]=str((base/name).resolve())
    return env
def job(label,command,scratch=None):
    record=R/(label+'-job.json')
    if record.exists():assert read(record).get('exit_code')==0,label;return
    deadline=time.monotonic()+1800
    while subprocess.check_output(['nvidia-smi','--id=4','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        state('waiting_for_gpu4',next_job=label)
        if time.monotonic()>deadline:raise RuntimeError('GPU4 occupied for30minutes; preserve experiment for continuation')
        time.sleep(30)
    if scratch and scratch.exists():
        assert scratch in [Path('data/repro_round005/benchmark_scratch'),Path('data/repro_round014/fresh_scratch'),Path('data/repro_round012/fresh_scratch')]
        shutil.rmtree(scratch)
    state('running',current_job=label)
    run([PY,'scripts/run_repro_job.py','--record',str(record),'--log',str(R/(label+'.log')),'--gpu','4','--']+command,env_for(scratch) if scratch else None)
def prepare():
    assert not (R/'experiment-plan.json').exists()
    old=read(OLD/'experiment-plan.json');data=read(R/'certified-training-data-manifest.json')
    assert data['combined_rows']==443 and data['support_gate_passed'] and data['heldout_parent_overlap']==0
    assert read(R/'fresh-gold-validation.json')['passed']==22
    assert read(R/'replay-loss-integration-check.json')['status']=='passed'
    assert read(R/'chosen-token-integration-check.json')['status']=='passed'
    assert read(R/'control-loss-integration-check.json')['control_native_ce_only_verified']
    models={a:old['evaluation_models'][a] for a in CONTROLS};models.update({a:f'outputs/repro_round014/{a}/final' for a in ARMS})
    frozen=set(data['sha256'])|set(read(R/'fresh-data-manifest.json')['sha256'])|set(read(OLD/'fresh-data-manifest.json')['sha256'])
    frozen.update(['data/splits/v3_81.json','artifacts/reproduction/round-002-2026-10-02/task-oracles.json','scripts/build_round014_fresh_tasks.py','scripts/train_round014_preferences.py','scripts/run_round014_experiment.py','scripts/finalize_round014_evidence.py','scripts/run_frozen_task_file.py','scripts/run_repro_job.py','scripts/compare_repro_evals.py','AgenticArxiv/rl/train_grpo.py','AgenticArxiv/rl/grpo_reward.py','AgenticArxiv/rl/reward.py','AgenticArxiv/rl/trl_compat.py','AgenticArxiv/benchmark/metrics.py','AgenticArxiv/agents/prompt_templates.py'])
    for path in models.values():
        if not path.startswith('outputs/repro_round014'):
            frozen.update(str(p) for p in Path(path).iterdir() if p.is_file())
    frozen.add('.venv-repro/lib/python3.10/site-packages/trl/trainer/dpo_trainer.py')
    frozen.update(['scripts/build_round014_real_errors.py','scripts/certify_round014_dataset.py','scripts/verify_round014_chosen_tokens.py','scripts/verify_round014_control_loss.py',str(R/'control-loss-integration-check.json'),'AgenticArxiv/rl/multiturn_env.py','AgenticArxiv/benchmark/semantic_oracle.py','AgenticArxiv/agents/base_agent.py',str(R/'replay-loss-integration-check.json'),str(R/'chosen-token-integration-check.json'),str(R/'negative-admission-plan.json'),str(R/'negative-admission-audit.json'),str(R/'independent-data-check.json'),str(R/'certified-training-data-manifest.json'),str(R.parent/'round-013-2026-10-05/rollout_traces.jsonl'),str(R.parent/'round-013-2026-10-05/collection-oracles.json')])
    reused={str(OLD/leaf(a,g)/f):sha(OLD/leaf(a,g)/f) for a in CONTROLS for g in ['dev','fresh-iid_test','fresh-ood_test'] for f in ['summary.json','traces.jsonl','execution_protocol.json']}
    plan={'registered_at_utc':stamp(),'mainline':'45 certified actual error decisions plus398broad expert replay: fixed sigmoidDPO+CE versus matched full chosen-only SFT,2seeds; original SFT and earlier CT are controls','arms':ARMS,'controls':CONTROLS,'evaluation_models':models,'pairs':PAIRS,'source_adapter_sha256':sha(SOURCE/'adapter_model.safetensors'),'sha256':{p:sha(p) for p in sorted(frozen)},'reused_evidence_sha256':reused,'loss_type':['sigmoid','sft'],'loss_weights':[1.0,1.0],'beta':.1,'seeds':[43,44],'max_steps':111,'num_train_epochs':1,'actual_training_rows':443,'real_preference_rows':45,'expert_replay_rows':398,'effective_batch':4,'batch_size':1,'gradient_accumulation_steps':4,'lr':5e-6,'max_length':4096,'dropout_disabled':True,'scheduler':'cosine','warmup_ratio':.03,'optimizer':'paged_adamw_8bit','reference':'Frozen initial SFT clone,392tensors checked before/after and saved ref verified.','matched_ce_implementation':'Both nativeDPOTrainer; control loss_types[sft]/weights[1],preference arm[sigmoid,sft]/weights[1,1]. Same token-mean CE,forward rows,batch/accumulation weighting; rejected/reference scores unused forcontrol learning.', 'dataset':'45train-only actual model errors plus398expert replay. Every real chosen suffix CPU-certified at same visible state; all443chosen rows/order/tokens/masks matched. Replay rows chosen=rejected and nativepreference gradient zero.','planned_new_benchmark_runs':822,'planned_reused_development_runs':270,'planned_new_training_prompt_exposures':1772,'evaluation':'Four new candidates oldDev24+Round012requests66=360development;seven models fresh22*3=462confirmation. Three controls270hash-locked developmentrecords reused.','session_prefix':'bench_r20261003_075529','evaluation_seed':45,'repeat':3,'prompt_policy':'baseline','evaluation_dtype':'bfloat16','checkpoint_selection':'Finaloneepoch111steps only; allfour models frozen before any Round014 inference.','selection_policy':'Fixed weight1, no coefficient grid, best-seed choice, test-specific rules or filtering. Full recipe chosen fromtrain-onlydataaudit andprior observeddevelopment; no current candidate outputs used. Fresh22before alltraining. Data validity andsupport thresholds preregistered.','success_rule':'Report both-seed contrasts vs matched new continuedSFT AND earlierCT/source SFT, aggregate/per-task strict success, terminal behavior and compute. Recovery vs degraded DPO alone is not algorithmic gain vs supervised control.','stop_rule':'Do not claim preference gain unless bothseeds exceed matched chosen-onlySFT ANDsource/earlierCT onfresh parameters with oldDev behavior retained. Preserve all outcomes; no coefficient orcase tuning.','limits':['Known caption world and observed heldout parents;22new parameter contracts,not new-world or wholly unseen-program generalization.','Realerrors fromone source samplingseed;45unique decisions35semantics21parents. Prompt is benchmark-style visible-state reprompt,notexactoriginalGRPOtokens. Chosen expert mostly alreadyseen bysource;111steps coverone443rowpass.','Equalsteps/promptbatch,not equalFLOPs; native combinedlossaddsCEcompute. OneSFTseed,two newtrainingseeds.']}
    write(R/'experiment-plan.json',plan);state('prepared');print('Registered Round014 before training or inference',flush=True)
def main():
    with (R/'controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        execute()
def execute():
    plan=read(R/'experiment-plan.json')
    for p,h in {**plan['sha256'],**plan['reused_evidence_sha256']}.items():assert sha(p)==h,p
    for a in ARMS:
        if (R/(a+'-candidate-freeze.json')).exists():
            frozen=read(R/(a+'-candidate-freeze.json'));assert sha(Path(frozen['model'])/'adapter_model.safetensors')==frozen['candidate_sha256'];continue
        model=Path(plan['evaluation_models'][a]).parent
        if not (R/(a+'-train-job.json')).exists():assert not model.exists(),model
        job(a+'-train',[PY,'-u','scripts/train_round014_preferences.py','--objective','dpo' if a.startswith('real') else 'sft','--seed',a[-2:],'--output',str(model)])
        manifest=read(model/'final/training_manifest.json');assert manifest['actual_steps']==111 and manifest['dataset_rows']==443
        assert manifest['dpo_loss_weights']==([1.,1.] if a.startswith('real') else [1.])
        shutil.copy2(model/'final/training_manifest.json',R/(a+'-training-manifest.json'))
        from safetensors.torch import load_file
        import torch
        before=load_file(str(SOURCE/'adapter_model.safetensors'));after=load_file(str(model/'final/adapter_model.safetensors'))
        assert set(before)==set(after)
        ref=load_file(str(model/'final/ref/adapter_model.safetensors'));assert set(before)==set(ref) and all(torch.equal(before[k],ref[k]) for k in before)
        changed=sum(not torch.equal(before[k],after[k]) for k in before);assert changed==392
        write(R/(a+'-candidate-freeze.json'),{'model':str(model/'final'),'candidate_sha256':sha(model/'final/adapter_model.safetensors'),'changed_tensors':changed,'frozen_at_utc':stamp(),'before_round014_model_evaluation':True,'reference_equal_sft_tensors':392})
    for a in CONTROLS:
        for g in ['dev','previous-iid_test','previous-ood_test']:
            target=R/leaf(a,g);source=OLD/leaf(a,g.replace('previous','fresh'))
            if not target.exists():shutil.copytree(source,target)
    for a in ARMS:
        cmd=[PY,'-u','-m','AgenticArxiv.benchmark.run_benchmark','--agents','regex','--backend','transformers','--model',plan['evaluation_models'][a],'--local-device','cuda','--local-dtype','bfloat16','--task-set','expanded','--split','data/splits/v3_81.json:dev','--offline','--snapshot','data/repro_20261002/snapshot_curated_v1.json','--repeat','3','--seed','45','--save-traces','--prefix',plan['session_prefix'],'--prompt-policy','baseline','--output',str(R/(a+'-dev'))]
        job(a+'-dev',cmd,Path('data/repro_round005/benchmark_scratch'))
        for split in ['iid_test','ood_test']:frozen_job(plan,a,split,'previous',OLD,Path('data/repro_round012/fresh_scratch'))
    for split in ['iid_test','ood_test']:
        for a in MODELS:frozen_job(plan,a,split,'fresh',R,Path('data/repro_round014/fresh_scratch'))
    for group in GROUPS:
        split=group if group=='dev' else group.split('-',1)[1]
        oracle='artifacts/reproduction/round-002-2026-10-02/task-oracles.json' if group=='dev' else str((OLD if group.startswith('previous') else R)/'fresh-oracles.json')
        for base,candidate in PAIRS:
            output=R/(base+'-vs-'+candidate+'-'+group+'.json')
            if not output.exists():run([PY,'scripts/compare_repro_evals.py','--base',str(R/leaf(base,group)),'--sft',str(R/leaf(candidate,group)),'--oracles',oracle,'--output',str(output),'--base_label',base,'--sft_label',candidate,'--training_description','Real trainerrorpairs plus398broadreplay,matched chosen-onlycontrol; '+('observeddevelopmentrequests' if group=='dev' or group.startswith('previous') else 'newparameters,sameknownpaperworld,observedparents'),'--evaluation_split',split])
    state('finalizing')
    run([PY,'scripts/finalize_round014_evidence.py'])
    state('completed');run([PY,'scripts/build_round014_report.py']);print('Round014 completed',flush=True)
def frozen_job(plan,a,split,prefix,data,scratch):
    cmd=[PY,'-u','scripts/run_frozen_task_file.py','--tasks',str(data/'fresh-tasks.json'),'--tasks_sha256',plan['sha256'][str(data/'fresh-tasks.json')],'--split',str(data/'fresh-splits.json')+':'+split,'--model',plan['evaluation_models'][a],'--snapshot','data/repro_20261002/snapshot_curated_v1.json','--output',str(R/(prefix+'-'+a+'-'+split)),'--prefix',plan['session_prefix'],'--seed','45','--repeat','3']
    job(prefix+'-'+a+'-'+split,cmd,scratch)
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='prepare':prepare()
    else:
        try:main()
        except BlockingIOError:raise SystemExit('Existing Round014 controller holds the lock; no duplicate run started.')
        except Exception as exc:state('failed_or_wait_timeout',error=str(exc));raise
