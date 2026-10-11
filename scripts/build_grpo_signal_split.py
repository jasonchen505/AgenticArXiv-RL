"""从冻结策略探测结果构建 GRPO 训练切分。

先对当前策略跑一次冻结探测（README 阶段3；`--lr 0.0` 不更新权重）：

    python -m AgenticArxiv.rl.train_grpo --model outputs/sft/final \\
        --split data/splits/v3_81.json:train --lr 0.0 --max_steps 408 \\
        --allow_zero_variance --no-verify --output_dir outputs/grpo_frozen_probe

再用本脚本按「每任务至少 2 个有组内奖励方差的 prompt 组」筛选训练任务：

    python -m scripts.build_grpo_signal_split \\
        --probe outputs/grpo_frozen_probe/reward_probe_summary.json \\
        --source-split data/splits/v3_81.json:train \\
        --model outputs/sft/final \\
        --output data/splits/v6_grpo_train.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_split_ref(ref: str) -> Tuple[Path, str]:
    file_part, _, name = ref.rpartition(":")
    if not file_part or not name:
        raise SystemExit(f"切分引用要写成 FILE:NAME，收到 {ref!r}")
    return Path(file_part), name


def load_eligible_tasks(ref: str) -> List[str]:
    path, name = parse_split_ref(ref)
    document = json.loads(path.read_text(encoding="utf-8"))
    split = document.get("split", {})
    if name not in split:
        raise SystemExit(f"{path} 里没有切分 {name!r}（可选：{sorted(split)}）")
    return list(split[name])


def informative_group_count(entry: Mapping[str, Any]) -> int:
    groups = int(entry.get("group_count") or 0)
    fraction = float(entry.get("informative_group_fraction") or 0.0)
    return int(round(fraction * groups))


def select_tasks(
    probe_tasks: Mapping[str, Mapping[str, Any]],
    eligible: Sequence[str],
    min_informative_groups: int,
    quality_tasks: Mapping[str, Mapping[str, Any]] | None = None,
) -> Dict[str, List[str]]:
    buckets: Dict[str, List[str]] = {
        "rl_train": [],
        "dropped_low_signal": [],
        "ceiling_control": [],
        "zero_variance_low_quality": [],
        "missing": [],
    }
    for task_id in eligible:
        entry = probe_tasks.get(task_id)
        if entry is None or int(entry.get("group_count") or 0) <= 0:
            buckets["missing"].append(task_id)
            continue
        informative = informative_group_count(entry)
        if informative >= min_informative_groups:
            buckets["rl_train"].append(task_id)
        elif informative == 0:
            # A constant negative/partial reward gives no relative advantage,
            # but is not a mastered task. Require full-score extrema, and when
            # supplied, independently recomputed strict success on every sample.
            extrema = [entry.get("min_reward"), entry.get("max_reward")]
            full_score = all(isinstance(v, (int, float)) and math.isfinite(v)
                             and v >= 1.0 - 1e-6 for v in extrema)
            if quality_tasks is not None:
                quality = quality_tasks.get(task_id, {})
                full_score = (full_score and quality.get("samples", 0) > 0
                              and quality.get("samples") == entry.get("sample_count")
                              and quality.get("strict_success_count") == quality.get("samples"))
            bucket = "ceiling_control" if full_score else "zero_variance_low_quality"
            buckets[bucket].append(task_id)
        else:
            buckets["dropped_low_signal"].append(task_id)
    return buckets


def audit_entry(entry: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(entry)
    merged["informative_group_count"] = informative_group_count(entry)
    return merged


def build_document(
    *,
    probe_path: Path,
    probe: Mapping[str, Any],
    source_ref: str,
    model: str,
    num_generations: int,
    seed: int,
    min_informative_groups: int,
    buckets: Mapping[str, List[str]],
) -> Dict[str, Any]:
    probe_tasks = probe["tasks"]
    selected = buckets["rl_train"]
    controls = buckets["ceiling_control"]
    dropped = buckets["dropped_low_signal"]
    low_quality = buckets.get("zero_variance_low_quality", [])
    audited = selected + controls + dropped + low_quality
    audit = {task_id: audit_entry(probe_tasks[task_id]) for task_id in audited}
    groups = sum(audit[task_id]["group_count"] for task_id in selected)
    informative = sum(
        audit[task_id]["informative_group_count"] for task_id in selected
    )
    groups_per_task = Counter(
        probe_tasks[task_id]["group_count"] for task_id in selected
    ).most_common(1)[0][0]
    return {
        "version": 7,
        "status": "probe_selected_pending_train",
        "task_set": "expanded",
        "purpose": (
            "setup-aware GRPO train split selected from frozen-SFT "
            "within-prompt reward variance"
        ),
        "derived_from": [source_ref, str(probe_path)],
        "source": {
            "model": model,
            "policy_frozen": True,
            "learning_rate": 0.0,
            "num_generations": num_generations,
            "groups_per_task": groups_per_task,
            "rollouts_per_task": groups_per_task * num_generations,
            "seed": seed,
            "reward_curriculum_steps": 0,
            "reward_summary": str(probe_path),
            "reward_summary_sha256": sha256_file(probe_path),
        },
        "selection_policy": {
            "eligibility": (
                f"Only tasks in {source_ref} are eligible; dev, iid_test and "
                "ood_test remain held out."
            ),
            "include": (
                f"Keep tasks with at least {min_informative_groups} informative "
                f"prompt groups out of {groups_per_task} in the frozen-policy probe."
            ),
            "exclude": (
                "Zero-variance full-score tasks form ceiling_control; constant "
                "failure/partial-credit or uncertified tasks form "
                "zero_variance_low_quality. Drop fewer-than-threshold "
                "informative groups as inconclusive. If strict quality is "
                "provided, all samples must pass to certify ceiling."
            ),
            "rationale": (
                "GRPO needs within-prompt reward variance. A task whose "
                "generations always receive the same reward contributes no "
                "relative-advantage gradient."
            ),
        },
        "split": {
            "rl_train": selected,
            "ceiling_control": controls,
            "zero_variance_low_quality": low_quality,
            "dropped_low_signal": dropped,
        },
        "audit": audit,
        "summary": {
            "probe_task_count": len(probe_tasks),
            "eligible_task_count": len(audited),
            "selected_task_count": len(selected),
            "ceiling_control_task_count": len(controls),
            "zero_variance_low_quality_task_count": len(low_quality),
            "dropped_task_count": len(dropped),
            "selected_prompt_group_count": groups,
            "selected_rollout_count": groups * num_generations,
            "informative_group_count": informative,
            "informative_group_fraction": (informative / groups) if groups else 0.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, help="reward_probe_summary.json 路径")
    parser.add_argument(
        "--source-split", required=True, help="候选任务来源，FILE:NAME（如 data/splits/v3_81.json:train）"
    )
    parser.add_argument("--model", required=True, help="被探测的策略模型路径")
    parser.add_argument("--output", required=True, help="输出切分文件路径")
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--min-informative-groups", type=int, default=2)
    parser.add_argument("--quality-summary", help="Independent strict probe summary from summarize_repro_probe.py")
    args = parser.parse_args()
    if args.min_informative_groups < 1 or args.num_generations < 2:
        raise SystemExit("Require positive informative-group threshold and at least two generations")

    probe_path = Path(args.probe)
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    if "tasks" not in probe:
        raise SystemExit(f"{probe_path} 里没有 tasks 字段，不是探测汇总文件")

    eligible = load_eligible_tasks(args.source_split)
    quality = None
    if args.quality_summary:
        quality_doc = json.loads(Path(args.quality_summary).read_text())
        if not quality_doc.get("adapter_tensors_unchanged"):
            raise SystemExit("Quality summary does not certify a frozen adapter")
        if quality_doc.get("reward_summary_sha256") != sha256_file(probe_path):
            raise SystemExit("Strict quality summary belongs to a different reward probe")
        adapter_path = Path(args.model) / "adapter_model.safetensors"
        if not adapter_path.exists() or quality_doc.get("source_adapter_sha256") != sha256_file(adapter_path):
            raise SystemExit("Strict quality summary belongs to a different policy adapter")
        quality = quality_doc["per_task"]
    buckets = select_tasks(probe["tasks"], eligible, args.min_informative_groups, quality)
    if buckets["missing"]:
        raise SystemExit(
            f"探测结果缺少 {len(buckets['missing'])} 个任务的统计："
            f"{buckets['missing'][:8]}（探测是否完全覆盖了 {args.source_split}？）"
        )
    if not buckets["rl_train"]:
        raise SystemExit(
            "探测没有选出任何有信号的任务；先确认探测的 lr 确实为 0、"
            "模型是当前策略，再考虑下调 --min-informative-groups"
        )

    document = build_document(
        probe_path=probe_path,
        probe=probe,
        source_ref=args.source_split,
        model=args.model,
        num_generations=args.num_generations,
        seed=args.seed,
        min_informative_groups=args.min_informative_groups,
        buckets=buckets,
    )
    document["source"]["strict_quality_summary"] = args.quality_summary
    document["source"]["strict_quality_summary_sha256"] = (sha256_file(Path(args.quality_summary)) if args.quality_summary else None)
    document["source"]["ceiling_certified_by"] = "reward_and_strict_oracle" if quality is not None else "reward_extrema_only"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = document["summary"]
    print(f"已写入 {output}")
    print(
        "  rl_train={selected_task_count} 条 / {selected_prompt_group_count} 组"
        "（有信息组 {informative_group_count}，占 {informative_group_fraction:.3f}）".format(
            **summary
        )
    )
    print(
        "  ceiling_control={ceiling_control_task_count} 条，"
        "零方差低质量={zero_variance_low_quality_task_count} 条，"
        "低信号丢弃={dropped_task_count} 条".format(**summary)
    )


if __name__ == "__main__":
    main()
