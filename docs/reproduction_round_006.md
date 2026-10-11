# Round006 — task-distribution comparison (2026-10-03 America/Los_Angeles)

同起点、各51步：完整51任务池的重复IID/OOD严格成功 54/66，高于7任务池的 51/66；SFT参考为 55/66。Dev分别为 SFT 10/24、7任务池 9/24、51任务池 12/24。 扩大任务池尚未超过SFT参考。

这是单种子、已观察测试集上的任务分布对照，不证明稳健或盲测泛化收益。 SFT remains the reference; no model promotion. This directly tests one proposed data-distribution change after Round005's negative GRPO result.

## Pre-registered comparison

Both arms start from identical full SFT1, seed43, lr1e-5, G4/batch4, temperature1, six turns,2048 completion, GRPO/group/token IS/beta.04/truncation mask, no curriculum/canary/automatic verifier. Both run51 optimizer steps with existing allow_zero_variance option, disabling sampled saturation early stop for equal budgets. No learner, reward or prompt edits. Seven-task arm controls the full51-task arm. Equal steps and rollouts, not equal informative groups, policy tokens or compute. One dataset changes sampling/RNG consumption; same seed does not pair identical randomness.

Pools use only existing train parents, with no held-out overlap. Full pool reduces infeasible-task prior from3/7 to3/51; observed rollout exposure is recorded below. This compares the whole distribution, not an isolated refusal-task ratio or guaranteed capability preservation. Zero reward variance can mean no relative-advantage signal; positive mean reward is not validation.

Both training arms use the same empty scratch path. Evaluation keeps baseline prompts,bf16,seed45,3 trials,matching session prefix,identical empty Round005 benchmark scratch path. Round005 SFT's90 runs are reused with locked hashes;180 new model evaluations executed. All comparisons independently rescored against frozen oracles,zero strict drift and exceptions. All final candidates frozen before Round006 evaluation,no within-round test tuning.

## Results

| Split | SFT reused | hard7,51steps | broad51,51steps |
| --- | --- | --- | --- |
| dev | 10/24 (pass³ 3/8) | 9/24 (pass³ 3/8) | 12/24 (pass³ 4/8) |
| iid_test | 43/54 (pass³ 14/18) | 39/54 (pass³ 13/18) | 42/54 (pass³ 14/18) |
| ood_test | 12/12 (pass³ 4/4) | 12/12 (pass³ 4/4) | 12/12 (pass³ 4/4) |

pass³ means all3 runs strictly correct,not pass@3. IID/OOD were observed in Round005 and are explicitly reused development/regression tests. These results do not establish blind confirmation; one training seed and small task sets remain limitations. Prior Round005 35-step result is context, not an equal-budget control for this experiment.

## Behavioral evidence

Between-arm strict-count changes occur in 2 tasks, archived in task-delta-audit.json. On this run, Dev search_kw_agentic_rl and IID search_kw_rag each change from hard7 0/3 to broad51 3/3; other task strict counts are unchanged. Hard7 trial0 prematurely FINISHes claiming a missing paper reference, although the request is keyword search. Broad51 calls search_arxiv_papers with query,max_results5,days30. This is a narrow recovery in keyword retrieval, not broad proof of improvement or an isolated causal test of infeasible-task ratio. SFT Dev is1/3 and IID3/3 on these tasks.

## Actual training and costs

- hard7: pool7, actual coverage7,204 rollouts/51 groups, informative26, clipped0, blocked88/204; process757.08s, Torch allocation11.61GiB, sampled device max23534MiB. Adapter826483c6ed1c7e07a62eb1976c7aa99a93afe9627e03c3be09f9903b070dc538, changed392tensors.
- broad51: pool51, actual coverage51,204 rollouts/51 groups, informative15, clipped0, blocked12/204; process862.42s, Torch allocation12.74GiB, sampled device max22124MiB. Adaptera57b9b333e51df42dbaeb210e6cf30e1885f2546f46c3b87177e49cbfa60a6d7, changed392tensors.

Only GPU4 used;5–7 reserved. GPU memory sampled every5seconds,not continuous physical peak. Training rollout summaries mix changing policies and are not frozen-model accuracy estimates.

## Mainline and next action

Completed reproducible data/environment -> full SFT -> real GRPO -> first IID/OOD -> controlled data-distribution ablation. Improvement still requires replicated gains and an independent evaluation design. Select the next experiment from these task-level outcomes: if broader data helps, replicate it and test on a new frozen set; if it only reduces regression or still fails, retain SFT and change one capability-preservation learning choice. Avoid another round centered on local logging repairs.

The world is curated real-paper extractive-caption tools. The author's missing VLM, original snapshot and weights are not recovered; do not claim exact original reproduction. No commits or pushes.
