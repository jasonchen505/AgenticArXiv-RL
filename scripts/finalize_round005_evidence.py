#!/usr/bin/env python3
"""Finalize the main experiment only after all planned paired runs complete."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT=Path('artifacts/reproduction/round-005-2026-10-03')

def read(name):return json.loads((ROOT/name).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def good(d):return d['completed'] and d['tool_accurate'] and d['arg_score']==1 and d['ref_score']==1 and d['parse_fail']==0 and d['tool_fail']==0 and d['terminal_semantics_accurate']

def main():
    assert read('completion.json')['state']=='completed'
    plan=read('experiment-plan.json');freeze=read('candidate-freeze.json');manifest=read('grpo-training-manifest.json')
    assert freeze['training_exit_code']==0 and freeze['changed_tensor_count']>0
    assert sha(Path(freeze['candidate_model'])/'adapter_model.safetensors')==freeze['candidate_adapter_sha256']
    world=json.loads(Path('data/splits/v3_81.json').read_text())['split']
    total=0;table=[];counts={m:{'dev':0,'iid_test':0,'ood_test':0} for m in ['SFT','GRPO']}
    summaries={}
    for split in ['dev','iid_test','ood_test']:
        comp=read(split+'-comparison.json')
        assert comp['evaluation_split']==split
        for name in ['SFT','GRPO']:
            label=name.lower()+'-'+split
            summary=read(label+'/summary.json');summaries[label]=summary
            assert summary['error_count']==0 and summary['sample_count']==3*len(world[split])
            assert {d['task_id'] for d in summary['details']}==set(world[split])
            assert len({(d['task_id'],d['trial']) for d in summary['details']})==summary['sample_count']
            assert all(d['trial'] in [0,1,2] for d in summary['details'])
            assert read(label+'-job.json')['exit_code']==0
            counts[name][split]=sum(good(d) for d in summary['details'])
            assert counts[name][split]==round(comp['models'][name]['strict_success_rate']*summary['sample_count'])
            assert comp['models'][name]['authoritative_rescore_drift']==0
            total+=summary['sample_count']
        b,g=comp['models']['SFT'],comp['models']['GRPO']
        table.append(f"| {split} | {counts['SFT'][split]}/{b['runs']} | {counts['GRPO'][split]}/{g['runs']} | {b['reliability']['pass^3']:.2%} -> {g['reliability']['pass^3']:.2%} | {b['false_finish_rate']:.2%} -> {g['false_finish_rate']:.2%} | {b['avg_tokens']:.1f} -> {g['avg_tokens']:.1f} |")
    assert total==180
    completion=read('completion.json');completion['actual_benchmark_runs']=total
    (ROOT/'completion.json').write_text(json.dumps(completion,indent=2)+'\n')
    heldout={m:counts[m]['iid_test']+counts[m]['ood_test'] for m in counts}
    if heldout['GRPO']>heldout['SFT']:
        conclusion=f"GRPO 在首次 IID/OOD 的合计严格成功由 {heldout['SFT']}/66 提高到 {heldout['GRPO']}/66，但 Dev 为 {counts['SFT']['dev']}/24 → {counts['GRPO']['dev']}/24；只有一次 RL seed，暂不能宣称稳健泛化收益。"
        classification='single_seed_heldout_improvement_requires_replication'
    elif heldout['GRPO']==heldout['SFT']:
        conclusion=f"本轮 GRPO 未带来独立测试净收益：IID/OOD 合计均为 {heldout['SFT']}/66；Dev 为 {counts['SFT']['dev']}/24 → {counts['GRPO']['dev']}/24。保留完整 SFT 作为基线，下一步改进训练配方与任务分布。"
        classification='no_heldout_net_gain'
    else:
        conclusion=f"本轮 GRPO 在独立测试上退化：IID/OOD 合计严格成功 {heldout['SFT']}/66 → {heldout['GRPO']}/66；Dev {counts['SFT']['dev']}/24 → {counts['GRPO']['dev']}/24。保留完整 SFT 基线；训练奖励饱和没有转化为泛化收益。"
        classification='heldout_regression'
    outcome={'classification':classification,'plain_chinese':conclusion,'strict_counts':counts,'heldout_strict_counts':heldout,'heldout_runs_per_model':66,'default_reference_model':freeze['source_model'],'candidate_kept_for_diagnostics':freeze['candidate_model'],'one_rl_seed_only':True,'no_test_tuning':True,'no_original_author_exact_reproduction_claim':True}
    (ROOT/'mainline-outcome.json').write_text(json.dumps(outcome,ensure_ascii=False,indent=2)+'\n')
    old=json.loads(Path('artifacts/reproduction/round-004-2026-10-03/baseline-dev/summary.json').read_text())
    current=summaries['sft-dev'];key=lambda d:(d['task_id'],d['trial'])
    before={key(d):good(d) for d in old['details']};after={key(d):good(d) for d in current['details']}
    assert before==after
    (ROOT/'baseline-reproduction-check.json').write_text(json.dumps({'strict_outcome_identity':'24/24','old_strict':sum(before.values()),'new_strict':sum(after.values())},indent=2)+'\n')
    training=read('training-outcome-audit.json');job=read('grpo64-job.json')
    text=f"""# Reproduction round 005 — 2026-10-03 (America/Los_Angeles)

## Mainline outcome

{conclusion}

This round directly runs broader real GRPO and the first paired Dev/IID/OOD model evaluation. No prompt search, core learner repair, commits or pushes. All GPU work used GPU 4; GPUs 5–7 remain reserved for others. This is a controlled reproduction in the new frozen caption world, not exact recovery of the author's missing VLM, snapshot and weights.

## Real GRPO update

Source full SFT1 adapter: {freeze['source_model']}.
Candidate: {freeze['candidate_model']}.
Requested64 optimizer steps, actually completed {freeze['actual_optimizer_steps']} with exit0; evaluated the actual-stop final adapter, step {freeze['evaluated_checkpoint_step']}. Stop reason {manifest['stop_reason']}. The five-group full-score window is local saturation, not proof of global convergence. All {freeze['changed_tensor_count']} adapter tensors differ from SFT.

Training: seven full-census signal tasks, lr1e-5, seed43, effective batch4, G4, temperature1, six turns, 2048 completion, explicit GRPO loss, group scaling, token importance sampling, beta.04, truncated-completion mask, no curriculum, quality-aware variance guard ON. No automatic canary or StageVerifier; quality comes from full task execution. The hard signal pool includes three blocked/infeasible tasks out of seven; its task distribution differs from the full train pool. This experiment tests the recipe, not an isolated selection/guard ablation.

Actual training rollouts {training['samples']}, informative groups {training['informative_groups']}/{training['groups']}, clipping {training['clipped_samples']}. These are changing-policy training samples, not final-model success estimates. Training process {job['elapsed_seconds']:.2f}s, Torch allocated peak {manifest['peak_vram_gib']:.2f}GiB, sampled device max {job['sampled_max_device_used_mib']}MiB (five-second sampling, not a continuous physical peak).

Source SHA256 {freeze['source_adapter_sha256']}.
Candidate SHA256 {freeze['candidate_adapter_sha256']}.

## Complete paired results

| Split | SFT strict | GRPO strict | pass^3 SFT -> GRPO | false_finish SFT -> GRPO | avg tokens SFT -> GRPO |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(table)}

All 180 planned task runs completed, zero exceptions, zero authoritative strict rescoring drift. Dev8×3, IID18×3, OOD4×3 per model; first IID/OOD model results in this project. Baseline Dev strict outcomes reproduce round004 in all24 trials.

The same default baseline prompt, bf16, seed45, frozen snapshot/split/oracles, session prefix and identical empty round-owned scratch are used. No contract_v1 engineering effect is mixed into the training comparison. Candidate weight hash was frozen before IID/OOD, and final model was selected by actual training stop rather than a Dev checkpoint search. No training/test overlap; no hyperparameter or prompt change based on this round's test output.

One source SFT seed, one RL seed, small test sets. pass^3 is all three trials successful, not pass@3. False_finish follows the repository's narrow definition; other failures can exist. After first results these tests have been observed; future work must disclose reuse or create another held-out set.

## Mainline progress and next experiment

The executed chain now covers pinned environment and frozen real-paper tool world, train-only SFT data, full597-step QLoRA SFT, real GRPO weights, and first independent paired IID/OOD evaluation. Original full VLM/RL reproduction remains unfinished. Do not present training saturation as a robust algorithm gain.

Next work should change one learning/data choice with the same source weights and budget, then replicate across seeds and preserve an independent evaluation protocol. A narrow hard-signal pool can change the refusal/action distribution; balancing task families and preserving SFT capabilities is a candidate direction, not a proven explanation from this single run. Avoid another round centered on local logging repairs.

Validation is actual training plus180 paired task executions, weight and input hashes, authoritative trace rescoring, comparison-helper same-baseline check, compile and git diff checks. No new broad unit-test campaign was run because the core learner was unchanged.
"""
    Path('docs/reproduction_round_005.md').write_text(text)
    (ROOT/'reproduction_round_005.md').write_text(text)
    source_paths=['scripts/run_round005_followup.py','scripts/finalize_round005_evidence.py','scripts/compare_repro_evals.py','scripts/run_repro_job.py','AgenticArxiv/rl/train_grpo.py','AgenticArxiv/rl/grpo_reward.py','AgenticArxiv/agents/prompt_templates.py','AgenticArxiv/agents/agent_engine.py','AgenticArxiv/agents/base_agent.py','AgenticArxiv/benchmark/runner.py','AgenticArxiv/benchmark/run_benchmark.py','AgenticArxiv/benchmark/metrics.py','docs/reproduction_round_005.md']
    final={'source_hashes':{p:sha(p) for p in source_paths},'allowed_gpu_indices':[4],'training_seed':43,'prompt_policy':'baseline','model_hashes':{'SFT':freeze['source_adapter_sha256'],'GRPO':freeze['candidate_adapter_sha256']},'actual_benchmark_runs':180,'no_test_tuning':True,'original_full_reproduction_complete':False,'patch_scope':'Cumulative source diff against HEAD49c101f; no evidence blobs'}
    (ROOT/'final-source-manifest.json').write_text(json.dumps(final,indent=2)+'\n')
    patch=subprocess.check_output(['git','diff','--binary','HEAD'])
    for path in subprocess.check_output(['git','ls-files','--others','--exclude-standard'],text=True).splitlines():
        if path.startswith(('scripts/','docs/','AgenticArxiv/tests/')) and Path(path).is_file():
            diff=subprocess.run(['git','diff','--no-index','--binary','--','/dev/null',path],stdout=subprocess.PIPE)
            assert diff.returncode in (0,1);patch+=diff.stdout
    (ROOT/'changes.patch').write_bytes(patch)
    subprocess.run(['git','diff','--check'],check=True)
    print(json.dumps(outcome,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
