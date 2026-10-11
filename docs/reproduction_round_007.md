# Round007 — seed replication and new frozen tests

2026-10-04 America/Los_Angeles

新冻结22任务：SFT 50/66；seed43难例池 48/66、完整池 48/66；seed44难例池 51/66、完整池 48/66。完整任务池相对难例池的优势没有在两个种子均保持。尚不能建立两个种子均超过SFT的收益。

新参数契约在模型输出前冻结，但基于已观察留出父任务、同一已知论文世界。原OOD分组有1/4完整工具顺序已在训练中出现；两个RL种子和22任务不足以证明广泛模板泛化。

## Protocol

Seed44 independently starts from the identical SFT1,hard7/broad51 train pools,51steps/204rollouts each. Same lr1e-5,G4/batch4,temperature1,6turns/2048completion,GRPO/group/tokenIS/beta.04/truncationmask,no curriculum,no sampled variance early stop,no prompt/reward/learner edits. Seed43 uses unchanged Round006 final models. One SFT seed42,two RL seeds43/44. Hashes frozen before new model output,no Dev checkpoint selection or test-based tuning.

New22 tasks are deterministic parameter recombinations of existing held-out parents:18 IID and4 OOD. They have no train-parent overlap and no exact tool/setup/terminal contract overlap with original81 or SFT137 semantic instances. Fresh gold22/22,same curated caption snapshot. Expected arguments and oracles remain outside agent inputs; existing BenchmarkRunner uses request and visible setup as before. No new evaluation dataset enters training.

All5 models execute fresh tests,330runs;seed44 Dev adds48runs. Existing Dev for SFT/seed43 reuses72hash-locked runs. Total378new/72reused,zero exceptions or strict authoritative rescore drift. Paired same baseline prompt,bf16,seed45,prefix and empty scratch. New task IDs define new shared per-task random streams. All final candidates frozen before evaluation. Semantic gold is not model success.

## Metrics

| Split | SFT | hard43 | broad43 | hard44 | broad44 |
| --- | --- | --- | --- | --- | --- |
| dev | 10/24 (pass³3/8) | 9/24 (pass³3/8) | 12/24 (pass³4/8) | 12/24 (pass³4/8) | 10/24 (pass³3/8) |
| fresh-iid_test | 38/54 (pass³12/18) | 36/54 (pass³12/18) | 36/54 (pass³12/18) | 39/54 (pass³13/18) | 36/54 (pass³12/18) |
| fresh-ood_test | 12/12 (pass³4/4) | 12/12 (pass³4/4) | 12/12 (pass³4/4) | 12/12 (pass³4/4) | 12/12 (pass³4/4) |

Fresh and previous tasks have different parameters/instructions,so do not compare their absolute rates as if task difficulty were identical. pass³ means3/3 strict success,not pass@3. Seeds share the sameSFT source and test set; counts are correlated and not hundreds of independent tasks. Structural program audit finds16/22fresh tasks share a full tool sequence with training,including1/4legacyOOD. Exact request/setup/argument contracts still differ. No tasks were removed based on this audit or model outcomes. Treat original OOD labels as legacy groups,not proof all programs are unseen. See template-scope-audit.json. Fresh tests become observed after this run; further tuning must disclose reuse.

## Narrow behavioral difference

Only1new task has different strict counts across models;the other21have identical strict-count outcomes,not necessarily identical outputs. The changed case requests paper9 from a4-paper session. SFT gives the correct invalid-index explanation2/3;hard44 does3/3. hard43/broad43 refuse with the wrong reason,claiming a missing arXivID0/3. broad44 calls download with ref9 and then claims completion0/3. The scoring criterion was unchanged and this is a real semantic difference,not a new regex reinterpretation. See behavior-audit.json. There is no broad tool-capability improvement in these results; next work should isolate termination-semantics learning while retaining the SFT reference and equal budgets.

## Resources and mainline

{
  "hard44": {
    "steps": 51,
    "samples": 204,
    "groups": 51,
    "task_coverage": 7,
    "pool_size": 7,
    "informative_groups": 31,
    "blocked_rollouts": 84,
    "clipped": 0,
    "process_seconds": 771.62,
    "torch_allocated_peak_gib": 11.596085548400879,
    "sampled_device_max_mib": 22802,
    "changed_tensors": 392,
    "candidate_sha256": "a9dd31cd8272006d801b271694372c0f9b05283ef80202dfd2c7e19c67375de0",
    "task_rollout_counts": {
      "infeasible_no_session": 28,
      "state_cache_before_dl": 28,
      "infeasible_zero_index": 28,
      "chain_ai5_read_then_summary": 32,
      "infeasible_unknown_id": 28,
      "chain_lg10_dl_tr_last": 28,
      "search_CR_7d_5": 32
    }
  },
  "broad44": {
    "steps": 51,
    "samples": 204,
    "groups": 51,
    "task_coverage": 51,
    "pool_size": 51,
    "informative_groups": 10,
    "blocked_rollouts": 12,
    "clipped": 0,
    "process_seconds": 852.11,
    "torch_allocated_peak_gib": 12.589744567871094,
    "sampled_device_max_mib": 17650,
    "changed_tensors": 392,
    "candidate_sha256": "486bba10624554b81eef6adba5bbbdf6800cf6a4dcbff5b44c044604bf3afc87",
    "task_rollout_counts": {
      "figure_cv5_ref3": 4,
      "search_CL_30d_10": 4,
      "search_LG_14d_1": 4,
      "summary_cl5_bullet250": 4,
      "multi_cr5_cache1": 4,
      "figure_cv5_ref1": 4,
      "summary_ai5_tldr60": 4,
      "search_CL_7d_5": 4,
      "constraint_download_only": 4,
      "summary_ro5_structured250": 4,
      "read_ai5_default": 4,
      "constraint_two_downloads_only": 4,
      "chain_cv5_cache_dl_tr_cache": 4,
      "state_ref_last_of_10": 4,
      "chain_ro5_dl_three": 4,
      "ref_ctrl_word_handover": 4,
      "state_ref_last_active": 4,
      "ref_stress_uncertainty_aware": 4,
      "infeasible_zero_index": 4,
      "opt_threads": 4,
      "figure_ro5_ref2": 4,
      "analyze_cv5_ref1_desc": 4,
      "state_cache_before_dl": 4,
      "ref_stress_state_across": 4,
      "chain_ai5_dl2_tr2": 4,
      "constraint_cache_only": 4,
      "chain_ai5_read_then_summary": 4,
      "search_kw_llm": 4,
      "search_CR_7d_5": 4,
      "read_cv5_method": 4,
      "constraint_search_only": 4,
      "opt_keep_dual": 4,
      "multi_lg10_dl3": 4,
      "search_RO_3d_8": 4,
      "state_ref_ordinal_cn": 4,
      "analyze_cv5_ref3_trend": 4,
      "summary_cv5_structured120": 4,
      "ref_ctrl_word_marionette": 4,
      "state_translate_active": 4,
      "opt_force_tr": 4,
      "infeasible_no_session": 4,
      "read_lg5_conclusion": 4,
      "search_LG_3d_10": 4,
      "infeasible_unknown_id": 4,
      "chain_lg10_dl_tr_last": 4,
      "ref_stress_image_restoration": 4,
      "search_AI_30d_25": 4,
      "analyze_ro5_ref2_axes": 4,
      "search_CV_14d_3": 4,
      "read_cl5_result": 4,
      "ref_ctrl_null_translate": 4
    }
  }
}

OnlyGPU4 used,GPUs5–7reserved. Torch allocation differs from5second sampled device usage; sampled max is not continuous physical peak. Installed TRL clones the initial SFT adapter as ref for this PEFT path. Saved references of all4GRPO models are CPU-verified392/392tensor-identical toSFT; this rules out a missing SFT reference,not insufficient KL regularization. See reference-policy-audit.json. Core-source hashes unchanged. No broad unit-test campaign; validation is real training and complete paired task execution,weight/input hashes,CPU gold,authoritative rescoring,compile/diff and HTML browser checks.

Mainline now covers reproduction environment/data,full597stepSFT,realGRPO,negative held-out result,controlled task-distribution ablation,and second-seed/new-input confirmation. RetainSFT reference until replicated useful gains support promotion. Choose one capability-preservation learner change if data-only recovery remains belowSFT; if confirmation fails,report instability before adding complexity. Original author's VLM/snapshot/weights remain missing; this is not exact author reproduction. No commits or pushes.
