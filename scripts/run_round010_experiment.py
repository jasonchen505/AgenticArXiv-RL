#!/usr/bin/env python3
"""Run matched DPO/continued-SFT seeds43/44 then complete paired evaluations."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil,subprocess,sys,time
R=Path('artifacts/reproduction/round-010-2026-10-04');OLD=Path('artifacts/reproduction/round-008-2026-10-04');PY='.venv-repro/bin/python'
SOURCE=Path('outputs/repro_20261003/sft_e1/final');ARMS=['ct43','dpo43','ct44','dpo44'];MODELS=['sft']+ARMS
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def stamp():return datetime.now(timezone.utc).isoformat()
def state(phase,**x):write(R/'controller-status.json',{'phase':phase,'updated_at_utc':stamp(),**x})
def run(cmd,env=None):print('Executing',cmd,flush=True);subprocess.run(cmd,env=env,check=True)
def env_for(base):
    env=os.environ.copy()
    for key,name in [('PDF_RAW_PATH','pdf_raw'),('PDF_TRANSLATED_PATH','pdf_translated'),('PDF_FIGURES_PATH','pdf_figures')]:env[key]=str((base/name).resolve())
    return env
def job(label,command,scratch=None):
    record=R/(label+'-job.json')
    if record.exists():assert read(record)['exit_code']==0,label;return
    deadline=time.monotonic()+1800
    while subprocess.check_output(['nvidia-smi','--id=4','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        state('waiting_for_gpu4',next_job=label)
        if time.monotonic()>deadline:raise RuntimeError('GPU4 occupied for30minutes; preserve experiment for continuation')
        time.sleep(30)
    if scratch and scratch.exists():shutil.rmtree(scratch)
    state('running',current_job=label)
    run([PY,'scripts/run_repro_job.py','--record',str(record),'--log',str(R/(label+'.log')),'--gpu','4','--']+command,env_for(scratch) if scratch else None)

def prepare():
    assert not (R/'experiment-plan.json').exists()
    data=read(R/'preference-data-manifest.json');assert data['pairs']==448 and data['heldout_parent_overlap']==0
    assert read(R/'fresh-gold-validation.json')['passed']==22
    models={'sft':str(SOURCE),**{a:f'outputs/repro_round010_v2/{a}/final' for a in ARMS}}
    frozen=set(data['sha256'])|set(read(R/'fresh-data-manifest.json')['sha256'])|set(read(OLD/'fresh-data-manifest.json')['sha256'])
    frozen.update(['data/splits/v3_81.json','artifacts/reproduction/round-002-2026-10-02/task-oracles.json',str(SOURCE/'adapter_model.safetensors'),'scripts/build_round010_preferences.py','scripts/build_round010_fresh_tasks.py','scripts/train_round010_preferences.py','scripts/run_round010_experiment.py','scripts/run_frozen_task_file.py','AgenticArxiv/rl/train_grpo.py','AgenticArxiv/rl/grpo_reward.py','AgenticArxiv/rl/reward.py','AgenticArxiv/benchmark/metrics.py','AgenticArxiv/agents/prompt_templates.py'])
    reused={str(OLD/leaf/f):sha(OLD/leaf/f) for leaf in ['sft-dev','fresh-sft-iid_test','fresh-sft-ood_test'] for f in ['summary.json','traces.jsonl','execution_protocol.json']}
    plan={'registered_at_utc':stamp(),'mainline':'Preference learning vs chosen-only continuation, matched SFT source, chosen samples,51optimizersteps,2seeds; frozen new parameter confirmation',
      'arms':ARMS,'evaluation_models':models,'source_adapter_sha256':sha(SOURCE/'adapter_model.safetensors'),'sha256':{p:sha(p) for p in sorted(frozen)},'reused_evidence_sha256':reused,
      'objective_control':'DPO sigmoid beta.1 vs completion-only SFT; same448prompts/chosen/order, DPO additionally sees synthetic rejected texts. Other training settings matched.','seeds':[43,44],'max_steps':51,'effective_batch':4,'batch_size':1,'gradient_accumulation_steps':4,'lr':5e-6,'max_length':4096,'dropout_disabled_both':True,'scheduler':'cosine','warmup_ratio':.03,'optimizer':'paged_adamw_8bit','reference':'DPO clones frozen initial SFT defaultadapter toref, exacttensors checked before/after',
      'planned_new_benchmark_runs':690,'planned_reused_development_runs':90,'planned_training_prompt_exposures':816,
      'evaluation':'Fourcandidates Dev24+Round008requests66=360new development runs;five models Round010new22*3=330new confirmation runs. ReuseSFT90development records.','session_prefix':'bench_r20261003_075529','evaluation_seed':45,'repeat':3,'prompt_policy':'baseline','evaluation_dtype':'bfloat16','checkpoint_selection':'Finalstep51only,allfourcandidatesfrozenbefore firstRound010inference;no output-basedtuning',
      'limits':['Known-papercaption world andobservedheldoutparents; newcontracts butnotnewworld/unseenprograms.','Synthetic negatives,expertchosen alreadyseenin originalSFT. No heldouttraces/papers/tasks inpreferences. Linguisticwrappers notnewsemantics.','Equal steps/effectivepromptbatch,not FLOPs or total completiontokens. OneSFTseed,two newobjective seeds.']}
    write(R/'experiment-plan.json',plan);state('prepared');print('Registered Round010 matched objectives before training or inference',flush=True)

def main():
    plan=read(R/'experiment-plan.json')
    for p,h in {**plan['sha256'],**plan['reused_evidence_sha256']}.items():assert sha(p)==h,p
    for a in ARMS:
        if (R/(a+'-candidate-freeze.json')).exists():continue
        model=Path(plan['evaluation_models'][a]).parent;assert not model.exists(),model
        objective='dpo' if a.startswith('dpo') else 'sft'
        job(a+'-train',[PY,'-u','scripts/train_round010_preferences.py','--objective',objective,'--seed',a[-2:],'--output',str(model)])
        manifest=read(model/'final/training_manifest.json');assert manifest['actual_steps']==51
        shutil.copy2(model/'final/training_manifest.json',R/(a+'-training-manifest.json'))
        from safetensors.torch import load_file
        import torch
        before=load_file(str(SOURCE/'adapter_model.safetensors'));after=load_file(str(model/'final/adapter_model.safetensors'));assert set(before)==set(after)
        changed=sum(not torch.equal(before[k],after[k]) for k in before);assert changed>0
        if objective=='dpo':
            ref=load_file(str(model/'final/ref/adapter_model.safetensors'));assert set(before)==set(ref) and all(torch.equal(before[k],ref[k]) for k in before)
        write(R/(a+'-candidate-freeze.json'),{'model':str(model/'final'),'candidate_sha256':sha(model/'final/adapter_model.safetensors'),'changed_tensors':changed,'frozen_at_utc':stamp(),'before_round010_model_evaluation':True,'reference_equal_sft_tensors':392 if objective=='dpo' else None})
    for src,target in [('sft-dev','sft-dev'),('fresh-sft-iid_test','previous-sft-iid_test'),('fresh-sft-ood_test','previous-sft-ood_test')]:
        if not (R/target).exists():shutil.copytree(OLD/src,R/target)
    for a in ARMS:
        cmd=[PY,'-u','-m','AgenticArxiv.benchmark.run_benchmark','--agents','regex','--backend','transformers','--model',plan['evaluation_models'][a],'--local-device','cuda','--local-dtype','bfloat16','--task-set','expanded','--split','data/splits/v3_81.json:dev','--offline','--snapshot','data/repro_20261002/snapshot_curated_v1.json','--repeat','3','--seed','45','--save-traces','--prefix',plan['session_prefix'],'--prompt-policy','baseline','--output',str(R/(a+'-dev'))]
        job(a+'-dev',cmd,Path('data/repro_round005/benchmark_scratch'))
        for split in ['iid_test','ood_test']:frozen_job(plan,a,split,'previous',OLD,Path('data/repro_round008/fresh_scratch'))
    for split in ['iid_test','ood_test']:
        for a in MODELS:frozen_job(plan,a,split,'fresh',R,Path('data/repro_round010/fresh_scratch'))
    pairs=[('sft',a) for a in ARMS]+[('ct43','dpo43'),('ct44','dpo44')]
    for group in ['dev','previous-iid_test','previous-ood_test','fresh-iid_test','fresh-ood_test']:
        split=group if group=='dev' else group.split('-',1)[1]
        oracle='artifacts/reproduction/round-002-2026-10-02/task-oracles.json' if group=='dev' else str((OLD if group.startswith('previous') else R)/'fresh-oracles.json')
        for base,candidate in pairs:
            leaf=lambda a:a+'-dev' if group=='dev' else group.split('-')[0]+'-'+a+'-'+split
            out=R/(base+'-vs-'+candidate+'-'+group+'.json')
            if not out.exists():run([PY,'scripts/compare_repro_evals.py','--base',str(R/leaf(base)),'--sft',str(R/leaf(candidate)),'--oracles',oracle,'--output',str(out),'--base_label',base,'--sft_label',candidate,'--training_description','Matched51stepDPOvscontinuedSFT; '+('observeddevelopmentrequests' if group=='dev' or group.startswith('previous') else 'newparameters,sameknownpaperworld,observedparents'),'--evaluation_split',split])
    write(R/'completion.json',{'state':'completed','planned_new_benchmark_runs':690,'planned_reused_development_runs':90,'actual_steps_per_arm':51,'no_within_round_test_tuning':True});state('completed')
    if Path('scripts/finalize_round010_evidence.py').exists():run([PY,'scripts/finalize_round010_evidence.py'])
    print('Round010 completed',flush=True)

def frozen_job(plan,a,split,prefix,data,scratch):
    cmd=[PY,'-u','scripts/run_frozen_task_file.py','--tasks',str(data/'fresh-tasks.json'),'--tasks_sha256',plan['sha256'][str(data/'fresh-tasks.json')],'--split',str(data/'fresh-splits.json')+':'+split,'--model',plan['evaluation_models'][a],'--snapshot','data/repro_20261002/snapshot_curated_v1.json','--output',str(R/(prefix+'-'+a+'-'+split)),'--prefix',plan['session_prefix'],'--seed','45','--repeat','3']
    job(prefix+'-'+a+'-'+split,cmd,scratch)
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='prepare':prepare()
    else:
        try:main()
        except Exception as exc:state('failed_or_wait_timeout',error=str(exc));raise
