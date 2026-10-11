#!/usr/bin/env python3
"""Finalize completed experiment metadata and cumulative reviewable source diff."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT=Path('artifacts/reproduction/round-004-2026-10-03')


def read(name):return json.loads((ROOT/name).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def good(d):return d['completed'] and d['tool_accurate'] and d['arg_score']==1 and d['ref_score']==1 and d['parse_fail']==0 and d['tool_fail']==0 and d['terminal_semantics_accurate']


def main():
    probe=read('full-probe-summary.json');comp=read('prompt-comparison.json')
    assert probe['samples']==408 and probe['tasks']==51 and probe['adapter_tensors_unchanged']
    for file in ['full-probe-v2-job.json','baseline-dev-job.json','contract-dev-job.json']:assert read(file)['exit_code']==0
    old=json.loads(Path('artifacts/reproduction/round-003-2026-10-03/sft-e1-dev/summary.json').read_text())
    new=read('baseline-dev/summary.json')
    key=lambda d:(d['task_id'],d['agent_type'],d['trial'])
    prior={key(d):good(d) for d in old['details']};current={key(d):good(d) for d in new['details']}
    assert set(prior)==set(current) and len(current)==24
    differences=[{'task_id':k[0],'agent_type':k[1],'trial':k[2],'previous_strict':prior[k],'current_strict':current[k]} for k in current if prior[k]!=current[k]]
    check={'old_strict_count':sum(prior.values()),'new_strict_count':sum(current.values()),'paired_runs':24,'strict_outcome_changes':len(differences),'outcome_identity':f'{24-len(differences)}/24 一致','differences':differences,'timing_identity_required':False}
    (ROOT/'baseline-reproduction-check.json').write_text(json.dumps(check,ensure_ascii=False,indent=2)+'\n')
    model='outputs/repro_20261003/sft_e1/final/adapter_model.safetensors'
    assert sha(model)==read('input-manifest.json')['sha256'][model]
    plan=read('prompt-ablation-plan.json')
    for path,expected in plan['candidate_source_hashes'].items():assert sha(path)==expected,path
    for name in ['SFT_BASELINE','SFT_CONTRACT_V1']:assert comp['models'][name]['authoritative_rescore_drift']==0
    splits=read('signal-split.json')['split'];faults=read('train-failures.json')
    b,c=comp['models']['SFT_BASELINE'],comp['models']['SFT_CONTRACT_V1']
    contract_summary=read('contract-dev/summary.json')
    assert new['sample_count']==contract_summary['sample_count']==24 and new['error_count']==contract_summary['error_count']==0
    arg_b=new['summary_by_agent']['regex']['arg_accuracy'];arg_c=contract_summary['summary_by_agent']['regex']['arg_accuracy']
    job=read('full-probe-v2-job.json')
    manifest=read('full-probe-training-manifest.json')
    counts={k:len(v) for k,v in splits.items()}
    text=f'''# Reproduction round 004 — 2026-10-03 (America/Los_Angeles)

All GPU work used GPU 4 only. GPUs 5–7 are reserved for others. No commits or pushes. No policy-weight update in this round. Original VLM/RL reproduction remains unfinished; the world is the curated extractive-caption snapshot from round002.

## Complete train-only frozen census

Fixed SFT1 adapter, 51 train parents, two groups/task, four generations/group, 408 actual rollouts / 102 optimizer-loop steps at lr=0. Temperature1, seed43, six turns, 2048 completion tokens, explicit GRPO loss, group reward scaling, token importance sampling, beta0.04, truncated-completion masking, no curriculum. The variance guard is explicitly disabled only for this frozen diagnostic, to guarantee complete coverage; no automatic canary or StageVerifier. Dev/IID/OOD do not enter selection. Saved/source adapter tensors are individually identical.

Strict success {probe['strict_success_count']}/408; informative groups {probe['informative_groups']}/102; clipped samples {probe['clipped_samples']}. Full process {job['elapsed_seconds']:.2f} seconds; device sampled max {job['sampled_max_device_used_mib']} MiB, not a continuous physical peak. Native GRPO and Benchmark differ in observation text, temperature and budgets; their success rates are not interchangeable. Two groups and one seed are limited evidence of variance, not proof of permanent mastery.

## Signal quality repair

The original split selector put every zero-variance task in ceiling_control. A consistent failure or partial-credit task is not a ceiling. New version separates full-score controls, zero_variance_low_quality, one-group/inconclusive signal and two-group signal. For this round, full-score controls additionally require strict success on every independently rescored sample. Quality input is SHA-bound to the original reward summary and policy adapter. Five new classification tests all failed before repair; two additional stale-evidence tests reject mismatched reward/policy hashes. Old round003 totals and three selected tasks are unchanged when reprocessed.

Task buckets: {json.dumps(counts)}. The actual census has no zero-variance low-quality task; the classification defect is reproduced by constructed regression inputs, not a measured membership change in this census. Sampling a different task set/order than round003 changes RNG consumption, so the same seed does not guarantee identical per-task generations; two-group membership is not a stability claim. Full-reward strict failures: {faults['full_reward_strict_failures']}; positive partial-reward strict failures: {faults['positive_reward_strict_failures']}. Fault categories overlap: {json.dumps(faults['fault_counts_overlapping'])}. These categories are audit evidence, not extra model inputs.

## One opt-in request-contract prompt ablation

Generic contract_v1 was fixed before any round004 Dev result. It is based on previous failure categories and current train diagnostics; prior Dev was already observed, so this is development, not a blind held-out test. It adds request/argument/context/finish checks without task IDs, oracle action lists, expected arguments, action rewriting or parameter postprocessing. Default prompts remain byte-identical. Explicit CLI flag --prompt-policy contract_v1 supports regex Agent only. Both variants use the same SFT1 weights.

Sequential fresh runs: same v3_81:dev eight tasks × three trials, seed45, bf16, matching prior session prefix, identical round-owned scratch cleared before each model. Source hashes and prompt text are frozen in prompt-ablation-plan.json. Round003 vs fresh baseline strict outcomes: {check['outcome_identity']}.

| Metric | Fresh SFT baseline | Same weights + contract_v1 |
| --- | --- | --- |
| Strict/pass^1 | {round(b['strict_success_rate']*24)}/24 ({b['strict_success_rate']:.2%}) | {round(c['strict_success_rate']*24)}/24 ({c['strict_success_rate']:.2%}) |
| pass^3 (all three succeed) | {b['reliability']['pass^3']:.2%} | {c['reliability']['pass^3']:.2%} |
| false_finish (repository definition) | {b['false_finish_rate']:.2%} | {c['false_finish_rate']:.2%} |
| Average parameter score (repository definition) | {arg_b:.2%} | {arg_c:.2%} |
| Average tokens | {b['avg_tokens']:.1f} | {c['avg_tokens']:.1f} |

Frozen-oracle strict rescoring drift is zero for both. Stable task gains {comp['task_gains']}, losses {comp['task_losses']}; descriptive paired p={comp['paired_task_exact_two_sided_p_descriptive']}. One candidate, one source training seed, eight development tasks. This is inference prompt engineering, not a new post-training gain; IID/OOD model tests remain unused. Success and added token cost must both be reported. The default policy remains baseline. The two additional successes occur only on search_kw_agentic_rl (1/3 to 3/3), where missing days=30 is now supplied. Average tokens increase {(c["avg_tokens"]/b["avg_tokens"]-1):.2%}, false_finish stays 25%. Force/save_to_file omissions and invented unsupported tools remain. The candidate additionally omits aspect on constraint_search_no_file, lowering that task parameter score from .75 to .5 despite the same strict failure. Report this limited Dev gain and tradeoff, not general argument-fidelity repair.

## Validation and reproducibility

Selected tests: 76 passed / eight subtests; compile and git diff checks passed. This is not the full repository suite. Preserve initial omitted-snapshot launch rejected before model load, the nonexistent-test-path invocation that ran no tests, and corrected summary field-name diagnostic. Final data are from the completed v2 launch; no partial data drive selection.

The report includes full per-task signal filtering, paired Dev results, complete train failure histories, logs, manifests, source hashes and a cumulative patch against the original source commit. Continue with the full signal pool, focused train-only data/RL changes, multiple seeds and untouched final tests before claiming robust algorithm improvement.
'''
    Path('docs/reproduction_round_004.md').write_text(text)
    (ROOT/'reproduction_round_004.md').write_text(text)
    paths=list(plan['candidate_source_hashes'])+['scripts/build_grpo_signal_split.py','scripts/summarize_repro_probe.py','scripts/audit_repro_rollout_failures.py','scripts/run_round004_followup.py','scripts/run_repro_job.py','AgenticArxiv/rl/train_grpo.py','AgenticArxiv/rl/grpo_reward.py','AgenticArxiv/tests/test_grpo_signal_quality.py','AgenticArxiv/tests/test_request_contract.py','docs/reproduction_round_004.md','scripts/finalize_round004_evidence.py']
    final={'source_hashes':{p:sha(p) for p in paths},'adapter_sha256_unchanged':sha(model),'allowed_gpu_indices':[4],'no_weight_updates':True,'default_prompt_policy':'baseline','candidate_fixed_before_round004_dev':True,'original_full_reproduction_complete':False,'patch_scope':'Cumulative source diff against HEAD 49c101f; no artifact blobs included'}
    (ROOT/'final-source-manifest.json').write_text(json.dumps(final,indent=2)+'\n')
    patch=subprocess.check_output(['git','diff','--binary','HEAD'])
    for name in subprocess.check_output(['git','ls-files','--others','--exclude-standard'],text=True).splitlines():
        if name.startswith(('scripts/','AgenticArxiv/tests/','docs/')) and Path(name).is_file():
            diff=subprocess.run(['git','diff','--no-index','--binary','--','/dev/null',name],stdout=subprocess.PIPE)
            assert diff.returncode in (0,1),name
            patch+=diff.stdout
    (ROOT/'changes.patch').write_bytes(patch)
    subprocess.run(['git','diff','--check'],check=True)
    print(json.dumps({'probe_strict':probe['strict_success_count'],'signal_buckets':counts,'baseline_strict':round(b['strict_success_rate']*24),'contract_strict':round(c['strict_success_rate']*24),'baseline_consistency':check['outcome_identity'],'patch_bytes':len(patch)},ensure_ascii=False))


if __name__=='__main__':main()
