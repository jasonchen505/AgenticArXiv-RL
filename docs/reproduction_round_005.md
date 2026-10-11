# Reproduction round 005 — 2026-10-03 (America/Los_Angeles)

## Mainline outcome

本轮 GRPO 在独立测试上退化：IID/OOD 合计严格成功 55/66 → 49/66；Dev 10/24 → 9/24。保留完整 SFT 基线；训练奖励饱和没有转化为泛化收益。

This round directly runs broader real GRPO and the first paired Dev/IID/OOD model evaluation. No prompt search, core learner repair, commits or pushes. All GPU work used GPU 4; GPUs 5–7 remain reserved for others. This is a controlled reproduction in the new frozen caption world, not exact recovery of the author's missing VLM, snapshot and weights.

## Real GRPO update

Source full SFT1 adapter: outputs/repro_20261003/sft_e1/final.
Candidate: outputs/repro_round005/grpo64/final.
Requested64 optimizer steps, actually completed 35 with exit0; evaluated the actual-stop final adapter, step 35. Stop reason reward_variance_window_saturated. The five-group full-score window is local saturation, not proof of global convergence. All 392 adapter tensors differ from SFT.

Training: seven full-census signal tasks, lr1e-5, seed43, effective batch4, G4, temperature1, six turns, 2048 completion, explicit GRPO loss, group scaling, token importance sampling, beta.04, truncated-completion mask, no curriculum, quality-aware variance guard ON. No automatic canary or StageVerifier; quality comes from full task execution. The hard signal pool includes three blocked/infeasible tasks out of seven; its task distribution differs from the full train pool. This experiment tests the recipe, not an isolated selection/guard ablation.

Actual training rollouts 140, informative groups 17/35, clipping 0. These are changing-policy training samples, not final-model success estimates. Training process 519.32s, Torch allocated peak 11.56GiB, sampled device max 23534MiB (five-second sampling, not a continuous physical peak).

Source SHA256 807090d928e088f7e83a35f8ef64920b2c3c5a91d661bd2b9b78c4e68eb634cc.
Candidate SHA256 80ef43d12dd42a8f129b76c2a87a0bce38434575ee7d33188b0ed1b9b98ab48e.

## Complete paired results

| Split | SFT strict | GRPO strict | pass^3 SFT -> GRPO | false_finish SFT -> GRPO | avg tokens SFT -> GRPO |
| --- | --- | --- | --- | --- | --- |
| dev | 10/24 | 9/24 | 37.50% -> 37.50% | 25.00% -> 37.50% | 4331.8 -> 4075.6 |
| iid_test | 43/54 | 37/54 | 77.78% -> 61.11% | 3.70% -> 18.52% | 5886.9 -> 5425.6 |
| ood_test | 12/12 | 12/12 | 100.00% -> 100.00% | 0.00% -> 0.00% | 9423.8 -> 9423.8 |

All 180 planned task runs completed, zero exceptions, zero authoritative strict rescoring drift. Dev8×3, IID18×3, OOD4×3 per model; first IID/OOD model results in this project. Baseline Dev strict outcomes reproduce round004 in all24 trials.

The same default baseline prompt, bf16, seed45, frozen snapshot/split/oracles, session prefix and identical empty round-owned scratch are used. No contract_v1 engineering effect is mixed into the training comparison. Candidate weight hash was frozen before IID/OOD, and final model was selected by actual training stop rather than a Dev checkpoint search. No training/test overlap; no hyperparameter or prompt change based on this round's test output.

One source SFT seed, one RL seed, small test sets. pass^3 is all three trials successful, not pass@3. False_finish follows the repository's narrow definition; other failures can exist. After first results these tests have been observed; future work must disclose reuse or create another held-out set.

## Mainline progress and next experiment

The executed chain now covers pinned environment and frozen real-paper tool world, train-only SFT data, full597-step QLoRA SFT, real GRPO weights, and first independent paired IID/OOD evaluation. Original full VLM/RL reproduction remains unfinished. Do not present training saturation as a robust algorithm gain.

Next work should change one learning/data choice with the same source weights and budget, then replicate across seeds and preserve an independent evaluation protocol. A narrow hard-signal pool can change the refusal/action distribution; balancing task families and preserving SFT capabilities is a candidate direction, not a proven explanation from this single run. Avoid another round centered on local logging repairs.

Validation is actual training plus180 paired task executions, weight and input hashes, authoritative trace rescoring, comparison-helper same-baseline check, compile and git diff checks. No new broad unit-test campaign was run because the core learner was unchanged.
