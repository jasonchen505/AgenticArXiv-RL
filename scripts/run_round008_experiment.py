#!/usr/bin/env python3
"""Two-seed training-only blocked-reason cap; fixed source, pool and budget."""
from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys, time
from datetime import datetime,timezone
R=Path('artifacts/reproduction/round-008-2026-10-04')
OLD6=Path('artifacts/reproduction/round-006-2026-10-03')
OLD7=Path('artifacts/reproduction/round-007-2026-10-04')
PY='.venv-repro/bin/python'
SOURCE=Path('outputs/repro_20261003/sft_e1/final')
ORACLES='artifacts/reproduction/round-002-2026-10-02/task-oracles.json'
ARMS=['gate43','gate44'];MODELS=['sft','hard43','gate43','hard44','gate44']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def stamp():return datetime.now(timezone.utc).isoformat()
def run(cmd,env=None):print('Executing',cmd,flush=True);subprocess.run(cmd,check=True,env=env)
def env_for(base):
    env=os.environ.copy()
    for key,name in [('PDF_RAW_PATH','pdf_raw'),('PDF_TRANSLATED_PATH','pdf_translated'),('PDF_FIGURES_PATH','pdf_figures')]:env[key]=str((base/name).resolve())
    return env
def status(phase,**kwargs):write(R/'controller-status.json',{'phase':phase,'updated_at_utc':stamp(),**kwargs})
def wait_free(label):
    deadline=time.monotonic()+1800
    while True:
        raw=subprocess.check_output(['nvidia-smi','--id=4','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip()
        if not raw:return
        status('waiting_for_gpu4',next_job=label,occupied_process_count=len(raw.splitlines()),policy='No shared GPU job launched; no processes interrupted; never GPU5-7')
        if time.monotonic()>deadline:raise RuntimeError('GPU4 remained occupied for 30 minutes; experiment preserved for explicit continuation')
        time.sleep(30)
def job(label,cmd,base):
    record=R/(label+'-job.json')
    if record.exists():
        assert read(record)['exit_code']==0,label
        return
    wait_free(label)
    if base.exists():shutil.rmtree(base)
    status('running',current_job=label)
    run([PY,'scripts/run_repro_job.py','--record',str(record),'--log',str(R/(label+'.log')),'--gpu','4','--']+cmd,env_for(base))

def prepare():
    assert not (R/'experiment-plan.json').exists()
    assert read(R/'fresh-gold-validation.json')['passed']==22
    hard=read(OLD7/'experiment-plan.json')['training_tasks']['hard44']
    assert len(hard)==7 and set(hard)<=set(read('data/splits/v3_81.json')['split']['train'])
    audit=read(R/'train-only-objective-audit.json')
    assert audit['arms']['hard43']['counts']['material_advantage_changed_groups']>0
    write(R/'training-splits.json',{'split':{a:hard for a in ARMS},'selection':'Reuse frozen Round004 seven-task train-only signal pool, unchanged; all benchmark cases excluded'})
    models={'sft':str(SOURCE),'hard43':'outputs/repro_round006/hard7/final','hard44':'outputs/repro_round007/hard44/final',**{a:f'outputs/repro_round008/{a}/final' for a in ARMS}}
    files=[str(SOURCE/'adapter_model.safetensors'),'data/repro_20261002/snapshot_curated_v1.json','data/repro_20261002/sft_train.jsonl','data/splits/v3_81.json',ORACLES,str(R/'training-splits.json'),str(R/'train-only-objective-audit.json')]
    files+=list(read(R/'fresh-data-manifest.json')['sha256'])
    files+=list(read(OLD7/'fresh-data-manifest.json')['sha256'])
    files+=['AgenticArxiv/rl/grpo_reward.py','AgenticArxiv/rl/train_grpo.py','AgenticArxiv/rl/reward.py','AgenticArxiv/benchmark/metrics.py','AgenticArxiv/agents/prompt_templates.py','scripts/run_frozen_task_file.py','scripts/run_round008_experiment.py','scripts/build_round008_fresh_tasks.py','AgenticArxiv/tests/test_blocked_reason_objective.py']
    reused={}
    for a in ['sft','hard43','hard44']:
        for group in ['dev','iid_test','ood_test']:
            path=OLD7/(a+'-dev' if group=='dev' else 'fresh-'+a+'-'+group)
            for f in ['summary.json','traces.jsonl','execution_protocol.json']:reused[str(path/f)]=sha(path/f)
    plan={'registered_at_utc':stamp(),'mainline':'Single objective comparison: cap positive reward for wrong blocked FINISH reason. Fixed seven-task pool, SFT source, 51 steps; two RL seeds.',
      'objective':'blocked_reason_v1','formula':'if FINISH and blocked terminal applicable and incorrect reason: min(legacy_reward,0); otherwise legacy reward. Evaluation scorer unchanged.',
      'pool_choice':'Existing train-only signal pool gives materially changed relative-advantage groups under archived train-only audit. No new task selection and no benchmark traces used in training.',
      'arms':ARMS,'training_tasks':{a:hard for a in ARMS},'evaluation_models':models,'reference_model_hashes':{a:sha(Path(models[a])/'adapter_model.safetensors') for a in ['sft','hard43','hard44']},
      'sha256':{f:sha(f) for f in sorted(set(files))},'reused_evidence_sha256':reused,
      'max_steps':51,'lr':1e-5,'seeds':[43,44],'num_generations':4,'batch_size':4,'temperature':1,'max_turns':6,'max_completion_length':2048,'loss_type':'grpo','scale_rewards':'group','importance_sampling_level':'token','beta':.04,'mask_truncated_completions':True,'reward_curriculum_steps':0,'allow_zero_variance':True,
      'session_prefix':'bench_r20261003_075529','evaluation_seed':45,'repeat':3,'prompt_policy':'baseline','evaluation_dtype':'bfloat16',
      'planned_new_benchmark_runs':510,'planned_reused_development_runs':270,'planned_training_rollouts':408,
      'evaluation_plan':'gate43/44 on exposed Dev24+Round007fresh66 each (180 new); all five frozen models on Round008fresh22*3 (330 new). Reuse prior SFT/hard43/hard44 Dev+Round007fresh (270).',
      'scratch_protocol':'Seed43 uses original Round006 training scratch; seed44 original Round007 training scratch. Development eval identical original paths; new confirmation identical Round008 path. Round-owned scratch reset before each job.',
      'checkpoint_selection':'Final step51 only, frozen both candidates before any Round008 inference. No test-based tuning or best-seed promotion.',
      'limits':['One SFT seed and two RL seeds. Shared frozen known-paper caption world, not original author VLM.','Round007 tests exposed, hence development. Round008 new parameter contracts frozen before model outputs, but from observed parents and familiar tool programs; not blind new-world evaluation.','Equal optimizer-step and rollout budget, not equal gradients/tokens/compute. Existing control runs reused, source-code legacy parity checked across 408 old train rows.']}
    write(R/'experiment-plan.json',plan);status('prepared');print('Round008 registered before training or inference',flush=True)

def main():
    plan=read(R/'experiment-plan.json')
    for p,h in {**plan['sha256'],**plan['reused_evidence_sha256']}.items():assert sha(p)==h,p
    for a,h in plan['reference_model_hashes'].items():assert sha(Path(plan['evaluation_models'][a])/'adapter_model.safetensors')==h
    for a in ARMS:
        seed=int(a[-2:]);model=Path(plan['evaluation_models'][a]).parent
        if not (R/(a+'-candidate-freeze.json')).exists():
            assert not model.exists(),model
            cmd=[PY,'-u','-m','AgenticArxiv.rl.train_grpo','--model',str(SOURCE),'--output_dir',str(model),'--task_set','expanded','--split',str(R/'training-splits.json')+':'+a,'--snapshot','data/repro_20261002/snapshot_curated_v1.json','--lr','1e-5','--max_steps','51','--batch_size','4','--num_generations','4','--max_completion_length','2048','--max_turns','6','--temperature','1','--reward_curriculum_steps','0','--reward_objective','blocked_reason_v1','--seed',str(seed),'--canary_steps','0','--no-verify','--save_rollout_traces','--report_to','none','--loss_type','grpo','--scale_rewards','group','--importance_sampling_level','token','--beta','.04','--mask_truncated_completions','--allow_zero_variance','--save_steps','17','--save_total_limit','2']
            job(a+'-train',cmd,Path('data')/('repro_round006' if seed==43 else 'repro_round007')/'training_scratch')
            m=read(model/'final/training_manifest.json');assert m['actual_steps']==51 and m['reward_objective']=='blocked_reason_v1'
            shutil.copy2(model/'final/training_manifest.json',R/(a+'-training-manifest.json'))
            for f,target in [('rollout_traces.jsonl','rollout-traces.jsonl'),('reward_probe_summary.json','reward-summary.json')]:shutil.copy2(model/f,R/(a+'-'+target))
            from safetensors.torch import load_file
            import torch
            before=load_file(str(SOURCE/'adapter_model.safetensors'));after=load_file(str(model/'final/adapter_model.safetensors'));ref=load_file(str(model/'final/ref/adapter_model.safetensors'))
            assert set(before)==set(after)==set(ref)
            changed=sum(not torch.equal(before[k],after[k]) for k in before);assert changed>0
            assert all(torch.equal(before[k],ref[k]) for k in before)
            write(R/(a+'-candidate-freeze.json'),{'model':str(model/'final'),'candidate_sha256':sha(model/'final/adapter_model.safetensors'),'source_sha256':sha(SOURCE/'adapter_model.safetensors'),'actual_steps':51,'changed_tensors':changed,'ref_equal_sft_tensors':len(ref),'frozen_at_utc':stamp(),'before_round008_model_evaluation':True})
    for a in ['sft','hard43','hard44']:
        for group in ['dev','iid_test','ood_test']:
            leaf=a+'-dev' if group=='dev' else 'previous-'+a+'-'+group
            src=OLD7/(a+'-dev' if group=='dev' else 'fresh-'+a+'-'+group)
            if not (R/leaf).exists():shutil.copytree(src,R/leaf)
    for a in ARMS:
        cmd=[PY,'-u','-m','AgenticArxiv.benchmark.run_benchmark','--agents','regex','--backend','transformers','--model',plan['evaluation_models'][a],'--local-device','cuda','--local-dtype','bfloat16','--task-set','expanded','--split','data/splits/v3_81.json:dev','--offline','--snapshot','data/repro_20261002/snapshot_curated_v1.json','--repeat','3','--seed','45','--save-traces','--prefix',plan['session_prefix'],'--prompt-policy','baseline','--output',str(R/(a+'-dev'))]
        job(a+'-dev',cmd,Path('data/repro_round005/benchmark_scratch'))
        for split in ['iid_test','ood_test']:frozen_job(plan,a,split,'previous',OLD7,Path('data/repro_round007/fresh_scratch'))
    for split in ['iid_test','ood_test']:
        for a in MODELS:frozen_job(plan,a,split,'fresh',R,Path('data/repro_round008/fresh_scratch'))
    pairs=[('sft',a) for a in ['hard43','gate43','hard44','gate44']]+[('hard43','gate43'),('hard44','gate44')]
    for group in ['dev','previous-iid_test','previous-ood_test','fresh-iid_test','fresh-ood_test']:
        oracle=ORACLES if group=='dev' else str((OLD7 if group.startswith('previous') else R)/'fresh-oracles.json')
        for base,candidate in pairs:
            leaf=lambda a:a+'-dev' if group=='dev' else group.split('-')[0]+'-'+a+'-'+group.split('-',1)[1]
            out=R/(base+'-vs-'+candidate+'-'+group+'.json')
            if not out.exists():run([PY,'scripts/compare_repro_evals.py','--base',str(R/leaf(base)),'--sft',str(R/leaf(candidate)),'--oracles',oracle,'--output',str(out),'--base_label',base,'--sft_label',candidate,'--training_description','Fixed hard7,51steps,seed43/44; training-only blocked-reason reward cap','--evaluation_split',group])
    write(R/'completion.json',{'state':'completed','planned_new_benchmark_runs':510,'planned_reused_development_runs':270,'actual_steps_per_arm':51,'no_within_round_test_tuning':True});status('completed');print('Round008 pipeline completed',flush=True)

def frozen_job(plan,arm,split,prefix,data,scratch):
    cmd=[PY,'-u','scripts/run_frozen_task_file.py','--tasks',str(data/'fresh-tasks.json'),'--tasks_sha256',plan['sha256'][str(data/'fresh-tasks.json')],'--split',str(data/'fresh-splits.json')+':'+split,'--model',plan['evaluation_models'][arm],'--snapshot','data/repro_20261002/snapshot_curated_v1.json','--output',str(R/(prefix+'-'+arm+'-'+split)),'--prefix',plan['session_prefix'],'--seed','45','--repeat','3']
    job(prefix+'-'+arm+'-'+split,cmd,scratch)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='prepare':prepare()
    else:
        try:main()
        except Exception as exc:status('failed_or_wait_timeout',error=str(exc));raise
