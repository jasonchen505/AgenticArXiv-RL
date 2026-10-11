#!/usr/bin/env python3
"""Mainline round: fixed GRPO candidate -> Dev -> first IID/OOD evaluation."""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT=Path('artifacts/reproduction/round-005-2026-10-03')
PYTHON='.venv-repro/bin/python'
SFT=Path('outputs/repro_20261003/sft_e1/final')
TRAIN=Path('outputs/repro_round005/grpo64')
ORACLES='artifacts/reproduction/round-002-2026-10-02/task-oracles.json'


def run(command,env=None):
    print('Executing:',command,flush=True)
    subprocess.run(command,check=True,env=env)


def sha(file):return hashlib.sha256(Path(file).read_bytes()).hexdigest()


def main():
    plan=json.loads((ROOT/'experiment-plan.json').read_text())
    assert plan['allowed_gpu_indices']==[4]
    while True:
        job=json.loads((ROOT/'grpo64-job.json').read_text())
        if job['state']!='running':break
        time.sleep(5)
    if job['exit_code']==0:
        model=TRAIN/'final'
        manifest=json.loads((model/'training_manifest.json').read_text())
        actual_steps=manifest['actual_steps'];evaluated_step=actual_steps
        shutil.copy2(model/'training_manifest.json',ROOT/'grpo-training-manifest.json')
    else:
        checkpoints=sorted((p for p in TRAIN.glob('checkpoint-*') if (p/'adapter_model.safetensors').exists() and (p/'adapter_config.json').exists()),key=lambda p:int(p.name.split('-')[-1]))
        if not checkpoints:raise RuntimeError('Training failed without a valid checkpoint; no candidate to evaluate')
        model=checkpoints[-1];evaluated_step=int(model.name.split('-')[-1])
        logs=[ast.literal_eval(m.group()) for m in re.finditer(r"\{'loss':[^\n\r]*?\}",(ROOT/'grpo64.log').read_text(errors='replace'))]
        actual_steps=len(logs)
    for source,target in [('rollout_traces.jsonl','grpo-rollout-traces.jsonl'),('reward_probe_summary.json','grpo-reward-summary.json')]:
        if (TRAIN/source).exists():shutil.copy2(TRAIN/source,ROOT/target)
    from safetensors.torch import load_file
    import torch
    before=load_file(str(SFT/'adapter_model.safetensors'));after=load_file(str(model/'adapter_model.safetensors'))
    assert set(before)==set(after)
    changed=sum(not torch.equal(before[k],after[k]) for k in before)
    assert changed>0,'No adapter update occurred; cannot label candidate trained GRPO'
    for path in [str(SFT/'adapter_model.safetensors'),'data/repro_20261002/snapshot_curated_v1.json','data/splits/v3_81.json','AgenticArxiv/rl/train_grpo.py','AgenticArxiv/rl/grpo_reward.py','AgenticArxiv/agents/prompt_templates.py','AgenticArxiv/benchmark/runner.py']:
        assert sha(path)==plan['sha256'][path],path
    freeze={'source_model':str(SFT),'candidate_model':str(model),'requested_steps':64,'actual_optimizer_steps':actual_steps,'evaluated_checkpoint_step':evaluated_step,'training_exit_code':job['exit_code'],'source_adapter_sha256':sha(SFT/'adapter_model.safetensors'),'candidate_adapter_sha256':sha(model/'adapter_model.safetensors'),'changed_tensor_count':changed,'training_seed':43,'training_task_count':7,'prompt_policy':'baseline','checkpoint_selection':'Final actual-stop model or last valid saved checkpoint on failure; no Dev-based checkpoint search','frozen_before_any_round005_iid_ood_result':True,'iid_ood_results_used_for_tuning':False,'allowed_gpu_indices':[4]}
    (ROOT/'candidate-freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
    scratch=Path('data/repro_round005/benchmark_scratch')
    env=os.environ.copy()
    for key,leaf in [('PDF_RAW_PATH','pdf_raw'),('PDF_TRANSLATED_PATH','pdf_translated'),('PDF_FIGURES_PATH','pdf_figures')]:
        env[key]=str((scratch/leaf).resolve())
    models={'SFT':SFT,'GRPO':model}
    for split in ['dev','iid_test','ood_test']:
        for name,path in models.items():
            label=name.lower()+'-'+split
            if scratch.exists():shutil.rmtree(scratch)
            cmd=[PYTHON,'-u','-m','AgenticArxiv.benchmark.run_benchmark','--agents','regex','--backend','transformers','--model',str(path),'--local-device','cuda','--local-dtype','bfloat16','--task-set','expanded','--split','data/splits/v3_81.json:'+split,'--offline','--snapshot','data/repro_20261002/snapshot_curated_v1.json','--repeat','3','--seed','45','--save-traces','--prefix',plan['session_prefix'],'--prompt-policy','baseline','--output',str(ROOT/label)]
            run([PYTHON,'scripts/run_repro_job.py','--record',str(ROOT/(label+'-job.json')),'--log',str(ROOT/(label+'.log')),'--gpu','4','--']+cmd,env)
        run([PYTHON,'scripts/compare_repro_evals.py','--base',str(ROOT/('sft-'+split)),'--sft',str(ROOT/('grpo-'+split)),'--oracles',ORACLES,'--output',str(ROOT/(split+'-comparison.json')),'--base_label','SFT','--sft_label','GRPO','--training_description','One RL seed; fixed seven-task GRPO recipe. Requested64, actual'+str(actual_steps)+', evaluated'+str(evaluated_step),'--evaluation_split',split])
    (ROOT/'completion.json').write_text(json.dumps({'state':'completed','splits':['dev','iid_test','ood_test'],'planned_benchmark_runs':180,'candidate_model':str(model),'no_test_tuning':True},indent=2)+'\n')
    print('Round005 real GRPO update and paired Dev/IID/OOD completed',flush=True)


if __name__=='__main__':main()
