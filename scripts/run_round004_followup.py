#!/usr/bin/env python3
"""Sequential work for the active round: train census, then one Dev prompt ablation."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path('artifacts/reproduction/round-004-2026-10-03')
PYTHON='.venv-repro/bin/python'
MODEL=Path('outputs/repro_20261003/sft_e1/final')
PROBE=Path('outputs/repro_round004/full_probe')


def run(command,env=None):
    print('Executing:',command,flush=True)
    subprocess.run(command,check=True,env=env)


def main():
    assert json.loads((ROOT/'gpu-policy.json').read_text())['allowed_gpu_indices']==[4]
    while True:
        job=json.loads((ROOT/'full-probe-v2-job.json').read_text())
        if job['state']!='running':break
        time.sleep(5)
    assert job['exit_code']==0,'Full census failed: do not select from partial data'
    inputs=json.loads((ROOT/'input-manifest.json').read_text())['sha256']
    for name in ['data/repro_20261002/snapshot_curated_v1.json','data/splits/v3_81.json',str(MODEL/'adapter_model.safetensors')]:
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==inputs[name],name
    for source,target in [('final/training_manifest.json','full-probe-training-manifest.json'),('rollout_traces.jsonl','full-probe-rollout-traces.jsonl'),('reward_probe_summary.json','full-probe-reward-summary.json')]:
        shutil.copy2(PROBE/source,ROOT/target)
    run([PYTHON,'scripts/summarize_repro_probe.py','--directory',str(PROBE),'--source_model',str(MODEL),'--oracles','artifacts/reproduction/round-002-2026-10-02/task-oracles.json','--split',str(ROOT/'train-probe-split.json'),'--output',str(ROOT/'full-probe-summary.json')])
    probe=json.loads((ROOT/'full-probe-summary.json').read_text())
    assert probe['samples']==408 and probe['groups']==102 and probe['tasks']==51
    run([PYTHON,'scripts/build_grpo_signal_split.py','--probe',str(PROBE/'reward_probe_summary.json'),'--source-split',str(ROOT/'train-probe-split.json')+':probe','--model',str(MODEL),'--output',str(ROOT/'signal-split.json'),'--seed','43','--num-generations','4','--min-informative-groups','2','--quality-summary',str(ROOT/'full-probe-summary.json')])
    scratch=Path('data/repro_round004/benchmark_scratch')
    env=os.environ.copy()
    for key,leaf in [('PDF_RAW_PATH','pdf_raw'),('PDF_TRANSLATED_PATH','pdf_translated'),('PDF_FIGURES_PATH','pdf_figures')]:
        env[key]=str((scratch/leaf).resolve())
    for policy,label in [('baseline','baseline-dev'),('contract_v1','contract-dev')]:
        if scratch.exists():shutil.rmtree(scratch)
        command=[PYTHON,'-u','-m','AgenticArxiv.benchmark.run_benchmark','--agents','regex','--backend','transformers','--model',str(MODEL),'--local-device','cuda','--local-dtype','bfloat16','--task-set','expanded','--split','data/splits/v3_81.json:dev','--offline','--snapshot','data/repro_20261002/snapshot_curated_v1.json','--repeat','3','--seed','45','--save-traces','--prefix','bench_r20261003_075529','--prompt-policy',policy,'--output',str(ROOT/label)]
        run([PYTHON,'scripts/run_repro_job.py','--record',str(ROOT/(label+'-job.json')),'--log',str(ROOT/(label+'.log')),'--gpu','4','--']+command,env)
    run([PYTHON,'scripts/compare_repro_evals.py','--base',str(ROOT/'baseline-dev'),'--sft',str(ROOT/'contract-dev'),'--oracles','artifacts/reproduction/round-002-2026-10-02/task-oracles.json','--output',str(ROOT/'prompt-comparison.json'),'--base_label','SFT_BASELINE','--sft_label','SFT_CONTRACT_V1','--training_description','One inference prompt ablation on identical SFT1 weights; no optimizer update'])
    print('Round004 full census and single prompt ablation completed',flush=True)


if __name__=='__main__':main()
