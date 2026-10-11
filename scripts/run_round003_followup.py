#!/usr/bin/env python3
"""Sequential follow-up for this active reproduction round; GPU 4 only."""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT=Path('artifacts/reproduction/round-003-2026-10-03')
PYTHON='.venv-repro/bin/python'
MODEL=Path('outputs/repro_20261003/sft_e1/final')


def run(command,env=None):
    print('Executing',command,flush=True)
    subprocess.run(command,check=True,env=env)


def main():
    assert json.loads((ROOT/'gpu-policy.json').read_text())['allowed_gpu_indices']==[4]
    while True:
        job=json.loads((ROOT/'sft-e1-job.json').read_text())
        if job['state']!='running':break
        time.sleep(5)
    assert job['exit_code']==0,'SFT failed; refusing to evaluate a partial model'
    verify=json.loads((MODEL/'verification_report.json').read_text())
    assert verify['passed'],'SFT format verification failed'
    manifest=json.loads((MODEL/'training_manifest.json').read_text())
    assert manifest['data_sha256']=='584dae5ac235cbeb7a5cf5461c3dabdb9d4875b3807dcaee883711613ab6012f'
    rows=[ast.literal_eval(m.group()) for m in re.finditer(r"\{'loss':[^\n\r]*?\}",(ROOT/'sft-e1.log').read_text(errors='replace'))]
    assert len(rows)==597 and rows[-1]['epoch']==1.0,'Epoch coverage incomplete'
    for source,target in [('training_manifest.json','sft-e1-training-manifest.json'),('verification_report.json','sft-e1-verification.json')]:
        shutil.copy2(MODEL/source,ROOT/target)
    provenance={'actual_optimizer_steps_from_step_logs':len(rows),'actual_epoch':rows[-1]['epoch'],'sample_visits':4776,'adapter_sha256':hashlib.sha256((MODEL/'adapter_model.safetensors').read_bytes()).hexdigest(),'source_model':manifest['base_model'],'training_data_sha256':manifest['data_sha256'],'source_hashes':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['AgenticArxiv/rl/train_sft.py','AgenticArxiv/rl/train_grpo.py','AgenticArxiv/rl/grpo_reward.py','AgenticArxiv/rl/reward.py']}}
    (ROOT/'sft-e1-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    scratch=Path('data/repro_20261002/base_scratch')
    if scratch.exists():shutil.rmtree(scratch)
    env=os.environ.copy()
    for key,leaf in [('PDF_RAW_PATH','pdf_raw'),('PDF_TRANSLATED_PATH','pdf_translated'),('PDF_FIGURES_PATH','pdf_figures')]:
        env[key]=str((scratch/leaf).resolve())
    prefix='bench_r20261003_075529'
    command=[PYTHON,'-u','-m','AgenticArxiv.benchmark.run_benchmark','--agents','regex','--backend','transformers','--model',str(MODEL),'--local-device','cuda','--local-dtype','bfloat16','--task-set','expanded','--split','data/splits/v3_81.json:dev','--offline','--snapshot','data/repro_20261002/snapshot_curated_v1.json','--repeat','3','--seed','45','--save-traces','--prefix',prefix,'--output',str(ROOT/'sft-e1-dev')]
    run([PYTHON,'scripts/run_repro_job.py','--record',str(ROOT/'dev-e1-job.json'),'--log',str(ROOT/'dev-e1.log'),'--gpu','4','--']+command,env)
    run([PYTHON,'scripts/compare_repro_evals.py','--base','artifacts/reproduction/round-002-2026-10-02/base-dev','--sft',str(ROOT/'sft-e1-dev'),'--oracles','artifacts/reproduction/round-002-2026-10-02/task-oracles.json','--output',str(ROOT/'paired-comparison.json'),'--sft_label','SFT_E1','--training_description','one full-epoch training run (597 optimizer steps)'])
    protocol=json.loads(Path('artifacts/reproduction/round-002-2026-10-02/evaluation-protocol.json').read_text())
    protocol['models']['SFT_E1']=str(MODEL);protocol['initial_rollout_artifact_root']=str(scratch)
    protocol['source_note']='Benchmark metric and inference paths retained; GRPO scoring-observation preservation fixed separately in round003.'
    protocol['description']='Base / SFT30 / SFT_E1 Dev budget comparison, one training seed; warmup and cosine horizon vary with total steps.'
    (ROOT/'evaluation-protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    scratch=Path('data/repro_20261003/probe_e1')
    for key,leaf in [('PDF_RAW_PATH','pdf_raw'),('PDF_TRANSLATED_PATH','pdf_translated'),('PDF_FIGURES_PATH','pdf_figures')]:
        env[key]=str((scratch/leaf).resolve())
    output=Path('outputs/repro_20261003/probe_e1_2048')
    command=[PYTHON,'-u','-m','AgenticArxiv.rl.train_grpo','--model',str(MODEL),'--output_dir',str(output),'--task_set','expanded','--split',str(ROOT/'train-probe-split.json')+':probe','--snapshot','data/repro_20261002/snapshot_curated_v1.json','--lr','0.0','--max_steps','16','--batch_size','4','--num_generations','4','--max_completion_length','2048','--max_turns','6','--temperature','1.0','--reward_curriculum_steps','0','--seed','43','--allow_zero_variance','--canary_steps','0','--no-verify','--save_rollout_traces','--report_to','none','--loss_type','grpo','--scale_rewards','group','--importance_sampling_level','token','--beta','0.04','--mask_truncated_completions']
    run([PYTHON,'scripts/run_repro_job.py','--record',str(ROOT/'probe-e1-job.json'),'--log',str(ROOT/'probe-e1.log'),'--gpu','4','--']+command,env)
    for source,target in [('final/training_manifest.json','probe-e1-training-manifest.json'),('rollout_traces.jsonl','probe-e1-rollout-traces.jsonl'),('reward_probe_summary.json','probe-e1-reward-summary.json')]:
        shutil.copy2(output/source,ROOT/target)
    run([PYTHON,'scripts/summarize_repro_probe.py','--directory',str(output),'--source_model',str(MODEL),'--oracles','artifacts/reproduction/round-002-2026-10-02/task-oracles.json','--split',str(ROOT/'train-probe-split.json'),'--output',str(ROOT/'probe-e1-summary.json')])
    probe=json.loads((ROOT/'probe-e1-summary.json').read_text())
    if probe['selected_task_ids']:
        run([PYTHON,'scripts/build_grpo_signal_split.py','--probe',str(output/'reward_probe_summary.json'),'--source-split',str(ROOT/'train-probe-split.json')+':probe','--model',str(MODEL),'--output',str(ROOT/'signal-split.json'),'--seed','43','--num-generations','4','--min-informative-groups','2'])
    else:
        (ROOT/'signal-eligibility.json').write_text(json.dumps({'eligible':False,'reason':'No predeclared task has two informative groups. Do not force a GRPO update from zero-variance groups.'},indent=2)+'\n')
    print('Round003 SFT, paired Dev and frozen probe completed.',flush=True)


if __name__=='__main__':main()
