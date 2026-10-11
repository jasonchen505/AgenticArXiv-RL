#!/usr/bin/env python3
"""Pre-registered, equal-budget task-distribution comparison; GPU4 only."""
from pathlib import Path
import collections
import hashlib
import json
import os
import shutil
import subprocess

R=Path('artifacts/reproduction/round-007-2026-10-04')
OLD=Path('artifacts/reproduction/round-006-2026-10-03')
PY='.venv-repro/bin/python'
SOURCE=Path('outputs/repro_20261003/sft_e1/final')
ORACLES='artifacts/reproduction/round-002-2026-10-02/task-oracles.json'
ARMS=['hard44','broad44']
SPLITS=['dev','iid_test','ood_test']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def run(cmd,env=None):
    print('Executing',cmd,flush=True)
    subprocess.run(cmd,check=True,env=env)
def env_for(leaf):
    env=os.environ.copy();base=Path('data/repro_round005/benchmark_scratch') if leaf=='benchmark_scratch' else Path('data/repro_round007')/leaf
    for key,name in [('PDF_RAW_PATH','pdf_raw'),('PDF_TRANSLATED_PATH','pdf_translated'),('PDF_FIGURES_PATH','pdf_figures')]:
        env[key]=str((base/name).resolve())
    return env
def job(label,command,env):
    run([PY,'scripts/run_repro_job.py','--record',str(R/(label+'-job.json')),'--log',str(R/(label+'.log')),'--gpu','4','--']+command,env)

def prepare():
    assert not (R/'experiment-plan.json').exists()
    previous=read(OLD/'experiment-plan.json')
    for p,h in previous['sha256'].items():assert sha(p)==h,p
    data=read(R/'fresh-data-manifest.json')
    assert read(R/'fresh-gold-validation.json')['passed']==22
    tasks={a:previous['training_tasks'][b] for a,b in [('hard44','hard7'),('broad44','broad51')]}
    write(R/'training-splits.json',{'split':tasks})
    models={'sft':str(SOURCE),'hard43':read(OLD/'hard7-candidate-freeze.json')['model'],'broad43':read(OLD/'broad51-candidate-freeze.json')['model'],**{a:f'outputs/repro_round007/{a}/final' for a in ARMS}}
    plan={**previous,'mainline':'Seed44 replication and first evaluation of frozen new parameter-recombination test, against seed43 and SFT',
      'seed':44,'arms':ARMS,'training_split':str(R/'training-splits.json'),'training_tasks':tasks,'output_models':{a:models[a] for a in ARMS},'evaluation_models':models,
      'planned_new_benchmark_runs':378,'planned_fresh_runs':330,'planned_new_dev_runs':48,'reused_dev_runs':72,
      'heldout_policy':'New22 task contracts frozen before Round007 models evaluated. Derived from previously observed held-out parents, same known-paper world. No output-based case selection or within-round tuning.',
      'baseline_evidence_reuse':'Reuse only Round006 Dev for SFT/hard43/broad43; all5 models execute new fresh tasks',
      'scratch_protocol':'Identical new training scratch for both seed44 arms; old benchmark scratch for reused Dev inputs; identical fresh scratch for all5 new-test evaluations',
      'two_rl_seeds':[43,44],'fresh_task_file':str(R/'fresh-tasks.json'),'fresh_oracles':str(R/'fresh-oracles.json'),'fresh_split':str(R/'fresh-splits.json')}
    plan['sha256']={**previous['sha256'],**data['sha256']}
    for f in [str(R/'training-splits.json'),'scripts/build_round007_fresh_tasks.py','scripts/run_frozen_task_file.py','scripts/run_round007_experiment.py']:
        plan['sha256'][f]=sha(f)
    plan['reference_model_hashes']={a:sha(Path(models[a])/'adapter_model.safetensors') for a in ['sft','hard43','broad43']}
    old_labels={'sft':'sft','hard43':'hard7','broad43':'broad51'}
    plan['dev_evidence_hashes']={f'{a}/{f}':sha(OLD/(b+'-dev')/f) for a,b in old_labels.items() for f in ['summary.json','traces.jsonl','execution_protocol.json']}
    plan['boundaries']=['Single SFT seed,two RL seeds. Fresh parameter recombinations,not new papers or broad unseen templates.','Observed old parents guided task design; no fresh model outputs used. New tests are exposed after this round.','Both51step budgets,not equal effective gradients/tokens/compute.']
    write(R/'experiment-plan.json',plan)
    print('Registered seed44 and fresh-test evaluation before model output',flush=True)

def main():
    plan=read(R/'experiment-plan.json')
    for p,h in plan['sha256'].items():assert sha(p)==h,p
    for a in ARMS:
        model=Path('outputs/repro_round007')/a
        assert not model.exists(),model
        cmd=[PY,'-u','-m','AgenticArxiv.rl.train_grpo','--model',str(SOURCE),'--output_dir',str(model),'--task_set','expanded','--split',str(R/'training-splits.json')+':'+a,'--snapshot','data/repro_20261002/snapshot_curated_v1.json','--lr','1e-5','--max_steps','51','--batch_size','4','--num_generations','4','--max_completion_length','2048','--max_turns','6','--temperature','1','--reward_curriculum_steps','0','--seed','44','--canary_steps','0','--no-verify','--save_rollout_traces','--report_to','none','--loss_type','grpo','--scale_rewards','group','--importance_sampling_level','token','--beta','.04','--mask_truncated_completions','--allow_zero_variance','--save_steps','17','--save_total_limit','2']
        scratch=Path('data/repro_round007/training_scratch')
        if scratch.exists():shutil.rmtree(scratch)
        job(a+'-train',cmd,env_for('training_scratch'))
        manifest=read(model/'final/training_manifest.json')
        assert manifest['actual_steps']==51 and not manifest['reward_variance_guard']['enabled']
        shutil.copy2(model/'final/training_manifest.json',R/(a+'-training-manifest.json'))
        for source,target in [('rollout_traces.jsonl','rollout-traces.jsonl'),('reward_probe_summary.json','reward-summary.json')]:
            shutil.copy2(model/source,R/(a+'-'+target))
        from safetensors.torch import load_file
        import torch
        before=load_file(str(SOURCE/'adapter_model.safetensors'));after=load_file(str(model/'final/adapter_model.safetensors'))
        assert set(before)==set(after)
        changed=sum(not torch.equal(before[k],after[k]) for k in before)
        assert changed>0
        write(R/(a+'-candidate-freeze.json'),{'model':str(model/'final'),'source_sha256':sha(SOURCE/'adapter_model.safetensors'),'candidate_sha256':sha(model/'final/adapter_model.safetensors'),'actual_steps':51,'changed_tensors':changed,'frozen_before_round006_evaluation':True})
    for p,h in plan['sha256'].items():assert sha(p)==h,p
    for a,h in plan['reference_model_hashes'].items():assert sha(Path(plan['evaluation_models'][a])/'adapter_model.safetensors')==h
    for a,b in [('sft','sft'),('hard43','hard7'),('broad43','broad51')]:
        for f in ['summary.json','traces.jsonl','execution_protocol.json']:assert sha(OLD/(b+'-dev')/f)==plan['dev_evidence_hashes'][f'{a}/{f}']
        shutil.copytree(OLD/(b+'-dev'),R/(a+'-dev'))
    for a in ARMS:
        scratch=Path('data/repro_round005/benchmark_scratch')
        if scratch.exists():shutil.rmtree(scratch)
        cmd=[PY,'-u','-m','AgenticArxiv.benchmark.run_benchmark','--agents','regex','--backend','transformers','--model',plan['output_models'][a],'--local-device','cuda','--local-dtype','bfloat16','--task-set','expanded','--split','data/splits/v3_81.json:dev','--offline','--snapshot','data/repro_20261002/snapshot_curated_v1.json','--repeat','3','--seed','45','--save-traces','--prefix',plan['session_prefix'],'--prompt-policy','baseline','--output',str(R/(a+'-dev'))]
        job(a+'-dev',cmd,env_for('benchmark_scratch'))
    pairs=[('sft',a) for a in ['hard43','broad43','hard44','broad44']]+[('hard43','broad43'),('hard44','broad44')]
    def compare(group,split,oracle):
        for base,candidate in pairs:
            directory=lambda a:R/((a+'-dev') if group=='dev' else ('fresh-'+a+'-'+split))
            run([PY,'scripts/compare_repro_evals.py','--base',str(directory(base)),'--sft',str(directory(candidate)),'--oracles',oracle,'--output',str(R/(base+'-vs-'+candidate+'-'+group+'.json')),'--base_label',base,'--sft_label',candidate,'--training_description','Seed43/44 matched51-step distributions; fresh test frozen before output; same paper world','--evaluation_split',split])
    compare('dev','dev',ORACLES)
    for split in ['iid_test','ood_test']:
        for a,model in plan['evaluation_models'].items():
            scratch=Path('data/repro_round007/fresh_scratch')
            if scratch.exists():shutil.rmtree(scratch)
            cmd=[PY,'-u','scripts/run_frozen_task_file.py','--tasks',plan['fresh_task_file'],'--tasks_sha256',plan['sha256'][plan['fresh_task_file']],'--split',plan['fresh_split']+':'+split,'--model',model,'--snapshot','data/repro_20261002/snapshot_curated_v1.json','--output',str(R/('fresh-'+a+'-'+split)),'--prefix',plan['session_prefix'],'--seed','45','--repeat','3']
            job('fresh-'+a+'-'+split,cmd,env_for('fresh_scratch'))
        compare('fresh-'+split,split,plan['fresh_oracles'])
    write(R/'completion.json',{'state':'completed','planned_new_benchmark_runs':378,'reused_dev_runs':72,'actual_steps_per_arm':51,'fresh_models':list(plan['evaluation_models']),'fresh_tasks':22,'no_within_round_test_tuning':True,'old_parent_design_disclosed':True})
    print('Round007 seed44 replication and frozen fresh test completed',flush=True)

if __name__=='__main__':
    import sys
    if len(sys.argv)>1 and sys.argv[1]=='prepare':prepare()
    else:main()
