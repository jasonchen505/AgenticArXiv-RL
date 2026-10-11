#!/usr/bin/env python3
"""Pre-registered, equal-budget task-distribution comparison; GPU4 only."""
from pathlib import Path
import collections
import hashlib
import json
import os
import shutil
import subprocess

R=Path('artifacts/reproduction/round-006-2026-10-03')
OLD=Path('artifacts/reproduction/round-005-2026-10-03')
PY='.venv-repro/bin/python'
SOURCE=Path('outputs/repro_20261003/sft_e1/final')
ORACLES='artifacts/reproduction/round-002-2026-10-02/task-oracles.json'
ARMS=['hard7','broad51']
SPLITS=['dev','iid_test','ood_test']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def run(cmd,env=None):
    print('Executing',cmd,flush=True)
    subprocess.run(cmd,check=True,env=env)
def env_for(leaf):
    env=os.environ.copy();base=Path('data/repro_round005/benchmark_scratch') if leaf=='benchmark_scratch' else Path('data/repro_round006')/leaf
    for key,name in [('PDF_RAW_PATH','pdf_raw'),('PDF_TRANSLATED_PATH','pdf_translated'),('PDF_FIGURES_PATH','pdf_figures')]:
        env[key]=str((base/name).resolve())
    return env
def job(label,command,env):
    run([PY,'scripts/run_repro_job.py','--record',str(R/(label+'-job.json')),'--log',str(R/(label+'.log')),'--gpu','4','--']+command,env)

def prepare():
    if (R/'experiment-plan.json').exists():raise RuntimeError('Refusing to overwrite registered experiment')
    R.mkdir(parents=True,exist_ok=True)
    previous=read(OLD/'experiment-plan.json')
    for p,h in previous['sha256'].items():assert sha(p)==h,p
    split=read('data/splits/v3_81.json')['split']
    hard=read('artifacts/reproduction/round-004-2026-10-03/signal-split.json')['split']['rl_train']
    assert len(hard)==7 and len(split['train'])==51 and set(hard)<=set(split['train'])
    assert not set(split['train'])&set(sum((split[k] for k in SPLITS),[]))
    write(R/'training-splits.json',{'split':{'hard7':hard,'broad51':split['train']},'selection':'Pre-existing full train pool versus previous signal pool; no selection from Round006 evaluations'})
    plan={**previous,'mainline':'Equal-budget task-distribution ablation, hard7 vs full train51 from identical SFT1; repeat evaluation disclosed',
          'arms':ARMS,'training_split':str(R/'training-splits.json'),'training_tasks':{'hard7':hard,'broad51':split['train']},
          'max_steps':51,'quality_aware_variance_guard':False,'allow_zero_variance':True,
          'guard_policy':'Disable sampled reward saturation stop for BOTH arms to ensure equal optimizer budget. Keep rollout audits, truncated masks and paired quality evaluation.',
          'output_models':{a:f'outputs/repro_round006/{a}/final' for a in ARMS},
          'checkpoint_selection':'Only step51 final of each pre-registered arm; no Dev-based selection; failure stops the experiment for diagnosis',
          'heldout_policy':'IID/OOD observed in Round005. Reused regression/development tests, not untouched confirmation. No within-round test tuning.',
          'planned_training_rollouts_per_arm':204,'planned_new_benchmark_runs':180,
          'baseline_evidence_reuse':'Reuse frozen Round005 SFT benchmark traces and metrics (90 runs); hashes locked before training.',
          'scratch_protocol':'Both arms use identical empty training scratch; eval uses identical empty Round005 benchmark scratch path to preserve baseline observation text.',
          'sole_between_arm_change':'Training task pool/distribution, including sampling order induced by dataset length. Same RNG seed is not paired identical RNG consumption.',
          'boundaries':['Controlled new extractive-caption world, not original author VLM reproduction.','One training seed. Existing IID/OOD observed; no blind generalization claim.','Equal optimizer-step/rollout budget, not equal informative gradient groups, tokens or wall time.']}
    plan.pop('output_model',None)
    plan['sha256'][str(R/'training-splits.json')]=sha(R/'training-splits.json')
    plan['sha256']['scripts/run_round006_experiment.py']=sha('scripts/run_round006_experiment.py')
    plan['baseline_hashes']={f'{s}/{f}':sha(OLD/('sft-'+s)/f) for s in SPLITS for f in ['summary.json','traces.jsonl','execution_protocol.json']}
    write(R/'experiment-plan.json',plan)
    print('Registered equal-budget experiment',flush=True)

def main():
    plan=read(R/'experiment-plan.json')
    for p,h in plan['sha256'].items():assert sha(p)==h,p
    for a in ARMS:
        model=Path('outputs/repro_round006')/a
        assert not model.exists(),model
        cmd=[PY,'-u','-m','AgenticArxiv.rl.train_grpo','--model',str(SOURCE),'--output_dir',str(model),'--task_set','expanded','--split',str(R/'training-splits.json')+':'+a,'--snapshot','data/repro_20261002/snapshot_curated_v1.json','--lr','1e-5','--max_steps','51','--batch_size','4','--num_generations','4','--max_completion_length','2048','--max_turns','6','--temperature','1','--reward_curriculum_steps','0','--seed','43','--canary_steps','0','--no-verify','--save_rollout_traces','--report_to','none','--loss_type','grpo','--scale_rewards','group','--importance_sampling_level','token','--beta','.04','--mask_truncated_completions','--allow_zero_variance','--save_steps','17','--save_total_limit','2']
        scratch=Path('data/repro_round006/training_scratch')
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
    for split in SPLITS:
        target=R/('sft-'+split)
        for f in ['summary.json','traces.jsonl','execution_protocol.json']:
            assert sha(OLD/('sft-'+split)/f)==plan['baseline_hashes'][f'{split}/{f}']
        shutil.copytree(OLD/('sft-'+split),target)
        for a in ARMS:
            scratch=Path('data/repro_round005/benchmark_scratch')
            if scratch.exists():shutil.rmtree(scratch)
            cmd=[PY,'-u','-m','AgenticArxiv.benchmark.run_benchmark','--agents','regex','--backend','transformers','--model',plan['output_models'][a],'--local-device','cuda','--local-dtype','bfloat16','--task-set','expanded','--split','data/splits/v3_81.json:'+split,'--offline','--snapshot','data/repro_20261002/snapshot_curated_v1.json','--repeat','3','--seed','45','--save-traces','--prefix',plan['session_prefix'],'--prompt-policy','baseline','--output',str(R/(a+'-'+split))]
            job(a+'-'+split,cmd,env_for('benchmark_scratch'))
        for base,candidate in [('sft','hard7'),('sft','broad51'),('hard7','broad51')]:
            run([PY,'scripts/compare_repro_evals.py','--base',str(R/(base+'-'+split)),'--sft',str(R/(candidate+'-'+split)),'--oracles',ORACLES,'--output',str(R/(base+'-vs-'+candidate+'-'+split+'.json')),'--base_label',base,'--sft_label',candidate,'--training_description','Round006 equal51-step task-distribution ablation; seed43; IID/OOD reused from Round005','--evaluation_split',split])
    write(R/'completion.json',{'state':'completed','planned_new_benchmark_runs':180,'reused_sft_runs':90,'arms':ARMS,'actual_steps_per_arm':51,'no_within_round_test_tuning':True,'test_reuse_disclosed':True})
    print('Round006 training and evaluation complete',flush=True)

if __name__=='__main__':
    import sys
    if len(sys.argv)>1 and sys.argv[1]=='prepare':prepare()
    else:main()
