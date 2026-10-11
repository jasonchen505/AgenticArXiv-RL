# Reproduction round 002 — 2026-10-02 (America/Los_Angeles)

Career focus: Agent algorithms, post-training, engineering and infrastructure. This round completes a new-world SFT pilot, not the original author's complete experiment.

## Executed evidence

- Independent `.venv-repro`, repo-local uv (system Snap uv cannot use `/data/home`); resolved dependencies frozen in `artifacts/reproduction/round-002-2026-10-02/requirements.lock.txt`. Torch 2.5.1+cu124 / Transformers 4.57.6 / TRL 0.29.1 / PEFT 0.17.1 / bitsandbytes 0.45.5.
- Built a new arXiv world: ten retrieval pools, 41 actual downloaded PDFs, 1,134 tool records. Used extractive captions, not the original FigureQA VLM. The original author's snapshot and weights remain unavailable.
- Raw world passed 79/81 declared gold trajectories. `read_cv5_method` lacked the requested method section; `analyze_ro5_ref2_axes` lacked figure 2. Preserved raw files and failures. Swapped CV positions 4/5 and RO positions 2/4 within the same pools, preserving paper sets and all content observations; did not use model outcomes for selection. Curated world passed 81/81 with reward 1.0 each. This is a resource-curated offline protocol, not live arXiv ranking.
- Curated snapshot SHA-256: `26d217a2c93492619665f440b8628f649914cbcbf2cb0f22a4b3ecb0200650eb`.
- Reproduced an SFT generator bug: setup established state without exposing it in training prompts. Added visible `initial_history` to each prompt while excluding setup from supervised actions. Tested explicit state, differing state for identical user text, and implicit dependency setup. This round uses the deterministic generator; it does not claim a complete audit of LLM expert execution.
- Selected checks: 129 passed, 148 subtests. Compile and whitespace checks passed. Not the entire project test suite.
- Train-only generation: 51 train parents, 86 parameter variants, 398 expert step rows, 4,776 distinct linguistic variants, 137 semantic task instances. Held-out parent overlap 0. Length median 2,016, p90 2,357, max 2,617; no truncation at 4,096.
- Dataset SHA-256: `584dae5ac235cbeb7a5cf5461c3dabdb9d4875b3807dcaee883711613ab6012f`.
- Actual GPU 4 QLoRA run: 30 optimizer steps, effective batch 8, seed 42, lr 1e-4, NF4, LoRA r=16/alpha=32. Loss 0.3205957, Trainer runtime 114.1 s; complete process elapsed 204 s. Saved adapter under `outputs/repro_20261002/sft_smoke_30/final`, stage parse-rate gate 100%.
- Adapter SHA-256: `9834e29a4acd6ce09244f58c89a1791c54c5de6ec2c6ea591ce1624521541d7c`.
- Peak Torch allocated memory 6.91 GiB. One observed nvidia-smi device-usage sample was 21,076 MiB (~20.6 GiB), not a measured maximum. Do not substitute allocated memory for physical device budget.

## Paired Dev evaluation

Base and SFT30 used the same curated world, v3_81 Dev split, bf16 Transformers inference, GPU 4, seed 45, three trials per task, matching session prefix, and reset empty round-owned scratch directory. QLoRA was used for training; both evaluations used bf16 inference. Frozen semantic oracles authoritatively rescored every trace with zero strict-outcome drift. See `evaluation-protocol.json` and `paired-comparison.json`.

| Metric | Base | SFT30 |
| --- | ---: | ---: |
| Strict / pass¹ | 1/24 (4.17%) | 8/24 (33.33%) |
| pass³ (all three trials succeed) | 0/8 | 2/8 (25%) |
| false_finish, repository definition | 16.67% | 12.5% |
| Mean tokens | 7,584.5 | 5,598.0 |
| Runtime exceptions | 0/24 | 0/24 |

Stable gains appear on `multi_cv3_dl1` and `state_dl_then_cache`. Parameter and boundary failures remain: missing `force`, numeric paper IDs, wrong keyword tool/arguments, and unsupported actions replaced with irrelevant tools plus FINISH. The narrow false_finish definition does not count all wrong FINISH results: SFT still strictly fails 16 of 24 runs.

Limits: eight Dev tasks, one short training run; descriptive paired task p=0.5 does not support statistical significance. No setup-fix training ablation, no IID/OOD evaluation, no GRPO run. No direct comparison with the original author's different world. Do not present 4,776 variants as independent tasks or format verification as task success.

## Next round

Increase SFT budget under the frozen protocol, probe train-only multi-rollout reward variance, then attempt GRPO. Keep Dev for development; use untouched test splits for final reporting. Focused improvements should separately test argument fidelity, supported-action boundaries, and false completion. Record independent training seeds and cost before claiming robust algorithm gains.

Report: `artifacts/reproduction/round-002-2026-10-02/report.html`. No commit or push performed. Other projects' GPU jobs were not modified.
