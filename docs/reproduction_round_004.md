# Reproduction round 004 — 2026-10-03 (America/Los_Angeles)

All GPU work used GPU 4 only. GPUs 5–7 are reserved for others. No commits or pushes. No policy-weight update in this round. Original VLM/RL reproduction remains unfinished; the world is the curated extractive-caption snapshot from round002.

## Complete train-only frozen census

Fixed SFT1 adapter, 51 train parents, two groups/task, four generations/group, 408 actual rollouts / 102 optimizer-loop steps at lr=0. Temperature1, seed43, six turns, 2048 completion tokens, explicit GRPO loss, group reward scaling, token importance sampling, beta0.04, truncated-completion masking, no curriculum. The variance guard is explicitly disabled only for this frozen diagnostic, to guarantee complete coverage; no automatic canary or StageVerifier. Dev/IID/OOD do not enter selection. Saved/source adapter tensors are individually identical.

Strict success 370/408; informative groups 26/102; clipped samples 0. Full process 1741.40 seconds; device sampled max 22712 MiB, not a continuous physical peak. Native GRPO and Benchmark differ in observation text, temperature and budgets; their success rates are not interchangeable. Two groups and one seed are limited evidence of variance, not proof of permanent mastery.

## Signal quality repair

The original split selector put every zero-variance task in ceiling_control. A consistent failure or partial-credit task is not a ceiling. New version separates full-score controls, zero_variance_low_quality, one-group/inconclusive signal and two-group signal. For this round, full-score controls additionally require strict success on every independently rescored sample. Quality input is SHA-bound to the original reward summary and policy adapter. Five new classification tests all failed before repair; two additional stale-evidence tests reject mismatched reward/policy hashes. Old round003 totals and three selected tasks are unchanged when reprocessed.

Task buckets: {"rl_train": 7, "ceiling_control": 32, "zero_variance_low_quality": 0, "dropped_low_signal": 12}. The actual census has no zero-variance low-quality task; the classification defect is reproduced by constructed regression inputs, not a measured membership change in this census. Sampling a different task set/order than round003 changes RNG consumption, so the same seed does not guarantee identical per-task generations; two-group membership is not a stability claim. Full-reward strict failures: 0; positive partial-reward strict failures: 13. Fault categories overlap: {"argument_mismatch": 22, "tool_execution_failure": 6, "reference_mismatch": 19, "blocked_reason_mismatch": 12, "tool_sequence_mismatch": 20, "false_finish": 15, "parse_failure": 1, "not_completed": 1}. These categories are audit evidence, not extra model inputs.

## One opt-in request-contract prompt ablation

Generic contract_v1 was fixed before any round004 Dev result. It is based on previous failure categories and current train diagnostics; prior Dev was already observed, so this is development, not a blind held-out test. It adds request/argument/context/finish checks without task IDs, oracle action lists, expected arguments, action rewriting or parameter postprocessing. Default prompts remain byte-identical. Explicit CLI flag --prompt-policy contract_v1 supports regex Agent only. Both variants use the same SFT1 weights.

Sequential fresh runs: same v3_81:dev eight tasks × three trials, seed45, bf16, matching prior session prefix, identical round-owned scratch cleared before each model. Source hashes and prompt text are frozen in prompt-ablation-plan.json. Round003 vs fresh baseline strict outcomes: 24/24 一致.

| Metric | Fresh SFT baseline | Same weights + contract_v1 |
| --- | --- | --- |
| Strict/pass^1 | 10/24 (41.67%) | 12/24 (50.00%) |
| pass^3 (all three succeed) | 37.50% | 50.00% |
| false_finish (repository definition) | 25.00% | 25.00% |
| Average parameter score (repository definition) | 62.85% | 62.50% |
| Average tokens | 4331.8 | 4735.7 |

Frozen-oracle strict rescoring drift is zero for both. Stable task gains 1, losses 0; descriptive paired p=1. One candidate, one source training seed, eight development tasks. This is inference prompt engineering, not a new post-training gain; IID/OOD model tests remain unused. Success and added token cost must both be reported. The default policy remains baseline. The two additional successes occur only on search_kw_agentic_rl (1/3 to 3/3), where missing days=30 is now supplied. Average tokens increase 9.32%, false_finish stays 25%. Force/save_to_file omissions and invented unsupported tools remain. The candidate additionally omits aspect on constraint_search_no_file, lowering that task parameter score from .75 to .5 despite the same strict failure. Report this limited Dev gain and tradeoff, not general argument-fidelity repair.

## Validation and reproducibility

Selected tests: 76 passed / eight subtests; compile and git diff checks passed. This is not the full repository suite. Preserve initial omitted-snapshot launch rejected before model load, the nonexistent-test-path invocation that ran no tests, and corrected summary field-name diagnostic. Final data are from the completed v2 launch; no partial data drive selection.

The report includes full per-task signal filtering, paired Dev results, complete train failure histories, logs, manifests, source hashes and a cumulative patch against the original source commit. Continue with the full signal pool, focused train-only data/RL changes, multiple seeds and untouched final tests before claiming robust algorithm improvement.
