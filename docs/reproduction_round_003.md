# Reproduction round 003 — 2026-10-03 (America/Los_Angeles)

This round completes one epoch of new-world SFT and executes a small native GRPO pilot. It does not recover the author's missing snapshot or weights; FigureQA remains extractive captions, not the original VLM. No commit or push performed.

## Resource policy

The user requires GPU 4 only. Earlier diagnostic probes on GPU 5/6 were terminated when this instruction arrived. Cancelled partial traces and SIGTERM records are preserved, not treated as paired experiments. All subsequent work was sequential on GPU 4. The launcher rejects any GPU index other than 4. Do not use GPU 5–7 even if they appear idle, or interfere with unrelated jobs on GPU 0–3.

## Completed SFT and Dev

Fixed inputs: independent pinned environment, Qwen2.5-1.5B-Instruct revision, curated caption world, split and 4,776-row train-only dataset from round002. Fresh base initialization, seed 42, effective batch 8, NF4 QLoRA r16/alpha32, lr 1e-4, cosine schedule, completion-only loss, max_length 4096. Changing the budget also changes proportional warmup and cosine horizons; this is a budget recipe comparison, not a pure single-variable ablation.

- Completed 597 optimizer steps / one epoch / all 4,776 rows. Counted input tokens: 10,011,814.
- Average training loss 0.05204575; Trainer 2,253.21 seconds, full process 2,353.52 seconds.
- Torch allocated peak 6.93 GiB; five-second device samples max 21,076 MiB, not a continuous physical peak.
- Format verifier parse rate 1.0; this is not task success.
- Adapter: `outputs/repro_20261003/sft_e1/final`, SHA256 `807090d928e088f7e83a35f8ef64920b2c3c5a91d661bd2b9b78c4e68eb634cc`.

| Frozen Dev metric | Base | SFT30 | SFT one epoch | GRPO checkpoint-8 |
| --- | --- | --- | --- | --- |
| Strict / pass^1 | 1/24 | 8/24 | 10/24 | 9/24 |
| pass^3 (all 3 trials succeed) | 0/8 | 2/8 | 3/8 | 3/8 |
| false_finish (repository definition) | 16.67% | 12.5% | 25% | 25% |
| Average tokens | 7,584.5 | 5,598.0 | 4,334.3 | 4,331.9 |

Same world, v3_81:dev, seed 45, three trials, bf16, matching session prefix and empty round-owned scratch. Frozen-oracle strict rescoring has zero drift; zero run exceptions. Only eight Dev tasks and one training seed. IID/OOD model tests remain unused. Failures still include omitted force/save_to_file arguments, paper-ID misinterpretation, invented unsupported tools and false completion. Lower training loss does not establish reliable task quality.

## Completion budget and corrected reward evidence

Environment observations have env_mask=0, but still consume the native completion budget. CPU execution of all 51 train gold paths shows 31 compact references exceed 256 tokens, four exceed 1,024, and max about 1,167. Compact gold excludes actual model Thought and some boundary costs, so this is not a guaranteed upper bound. The later native probe uses six turns / 2,048 tokens. A cancelled 256-token sample generated 62 policy tokens and filled its remaining budget with observations; incomplete 256/1,024 runs are diagnostics only.

Before repair, all 51 native gold paths were strict-correct, but a valid long FigureQA return for analyze_ro5_ref2_axes received -0.75 reward. Rollout truncated its repr to 1,000 characters before scoring, making the complete-answer parser reject malformed evidence. The repair keeps full history.observation for reward/audit and bounded model_observation for model input. RewardCalculator and its empty-answer gate are unchanged. After repair, all 51 gold paths remain strict-correct and each has reward 1.0. Two of three added tests failed before the patch. This proves label correction, not a learning-effect ablation.

## Frozen probe and actual GRPO pilot

Installed TRL 0.29.1 defaults to DAPO loss and no truncated-completion masking. Both the probe and pilot explicitly use loss_type=grpo, group reward scaling, token importance sampling, beta 0.04, truncated-completion masking, seed 43, temperature 1, G=4, no curriculum or automatic canary. The frozen probe uses lr=0, two groups per predeclared train task. Its saved/source adapter tensors are exactly identical.

Probe: 64 rollouts / 16 groups / eight train tasks, 53 strict successes, eight informative groups, zero clipped samples. Three tasks had reward differences in both groups: chain_lg10_dl_tr_last, infeasible_no_session, analyze_cv5_ref1_desc. This is not a census of all 51 train tasks. Native GRPO and Benchmark serialize observations differently, so their success rates are not interchangeable.

The actual pilot trains these three tasks at lr=1e-5. Requested 12 steps; completed nine before the original variance guard stopped with exit 1. All 36 raw rollouts are archived. Steps 5–9 each had full reward 1 and zero variance, but the old guard classified this early window as startup failure. Saved checkpoint-8 is evaluated; step-9 weights were not saved. All 392 adapter tensors differ from the SFT source. Device sampled max 24,210 MiB; Torch allocation peak is unavailable because the final manifest was not produced.

Evaluated adapter `outputs/repro_20261003/grpo12/checkpoint-8`, SHA256 `dcf354d7c1b286a330ee4938ce32650176cd297020ca4bc2c0349b1138cf8746`. Its Dev result is 9/24 versus SFT's 10/24, with unchanged pass^3 and false_finish. This pilot shows no Dev gain. Preserve it as a diagnostic and retain the full SFT adapter as preferred baseline.

## Quality-aware variance guard and validation

The guard now distinguishes an entire full-score, error-free zero-variance window from a low-quality window, independent of training step. Informative groups reset the streaks. Five added tests cover early ceiling, early/late failure, mixed window and reset; three failed before repair. Replaying the actual pilot logs yields converged=True, tripped=False at step nine. This is a counterfactual log replay, not a training rerun or newly saved model.

Final selected checks: 164 passed, 53 subtests passed, one skipped for the missing original default snapshot; compile and diff checks passed. An earlier CUDA-hidden attempt hit an existing BF16 configuration test unsupported on CPU; that log is preserved. The selected suite was then run with GPU 4 visible and passed. The new-world gold execution is separate evidence, not the skipped original-snapshot test.

## Next work and career claims

Expand the frozen train signal census, then test broader RL task coverage and parameter-fidelity / unsupported-action handling with controlled changes, multiple seeds and untouched final IID/OOD tests. The original full VLM/RL reproduction is unfinished. Interview claims can cover personally executed QLoRA training, trace-level budget/reward diagnosis, explicit loss configuration, quality-aware training guards, monitored cost and an honestly reported negative GRPO result. Do not claim the author's historical metrics or robust RL gains.
