#!/usr/bin/env python3
"""Recompute archived evaluation metrics; never generates model responses."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


def wilson(successes, n):
    if not n:
        return None
    z = 1.959963984540054
    p = successes / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [center - radius, center + radius]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, help='Original frozen snapshot for authoritative trace rescoring')
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / 'AgenticArxiv'))
    from benchmark.metrics import TaskMetrics, extract_metrics, is_strict_success
    from benchmark.report import BenchmarkReport
    from benchmark.tasks_expanded import get_expanded_tasks
    tasks = {t['id']: t for t in get_expanded_tasks()}
    if args.snapshot:
        from benchmark.semantic_oracle import attach_expected_paper_ids
        tasks = {t['id']: t for t in attach_expected_paper_ids(tasks.values(), args.snapshot)}
    prefixes = {
        'SFT': 'eval_sft_',
        'GRPO': 'eval_grpo_',
        'GSPO': 'eval_grpo_gspo_',
        'DrGRPO': 'eval_grpo_drgrpo_',
        'DAPO': 'eval_grpo_dapo_',
        'SFT-T5': 'eval_sft_t5_',
    }
    suffixes = {
        'rl_train': 'v6_rl_train_r3_s45',
        'dev': 'v381_dev_r3_s45',
        'iid_test': 'v381_iid_r3_s45',
        'ood_test': 'v381_ood_r3_s45',
    }
    evaluations = []
    decisions = {}
    for model, prefix in prefixes.items():
        for split, suffix in suffixes.items():
            folder = repo / 'artifacts' / (prefix + suffix)
            summary_path = folder / 'summary.json'
            if not summary_path.exists():
                raise FileNotFoundError(summary_path)
            raw = json.loads(summary_path.read_text())
            metrics = []
            for d in raw['details']:
                metrics.append(TaskMetrics(
                    task_id=d['task_id'], agent_type=d['agent_type'], trial=d['trial'],
                    session_id=d.get('session_id', ''), task_completed=d['completed'],
                    tool_call_accurate=d['tool_accurate'], arg_score=d['arg_score'],
                    ref_score=d['ref_score'], parse_failures=d['parse_fail'],
                    tool_exec_failures=d['tool_fail'],
                    terminal_semantics_accurate=d.get('terminal_semantics_accurate', True),
                    false_finish=d['false_finish'], error=d.get('error'),
                ))
            reliability = BenchmarkReport(metrics).reliability_by_agent()['regex']
            by_task = defaultdict(list)
            by_key = {}
            for m in metrics:
                good = is_strict_success(m)
                by_task[m.task_id].append(good)
                key = (m.task_id, m.agent_type, m.trial)
                if key in by_key:
                    raise ValueError(f'Duplicate task/trial: {folder}: {key}')
                by_key[key] = good
            if any(len(v) != 3 for v in by_task.values()):
                raise ValueError(f'Expected exactly three trials per task: {folder}')
            decisions[(model, split)] = {t: all(v) for t, v in by_task.items()}
            # Without the original snapshot, paper-id oracles cannot be restored.
            # Such trace comparisons are diagnostic, never metric reproduction.
            drift = []
            trace_count = 0
            seen = set()
            trace_path = folder / 'traces.jsonl'
            for line in trace_path.read_text().splitlines():
                trace = json.loads(line)
                key = (trace['task_id'], trace['agent_type'], trace['trial'])
                if key in seen:
                    raise ValueError(f'Duplicate archived trace: {folder}: {key}')
                seen.add(key)
                score = extract_metrics(tasks[trace['task_id']], trace,
                                        trace['agent_type'], trace['trial'])
                if is_strict_success(score) != by_key[key]:
                    drift.append({'task_id': key[0], 'trial': key[2],
                                  'stored': by_key[key], 'rescored': is_strict_success(score)})
                trace_count += 1
            if seen != set(by_key):
                raise ValueError(f'Trace/summary coverage mismatch: {folder}')
            fully_successful = sum(all(v) for v in by_task.values())
            evaluations.append({
                'model': model, 'split': split, 'source': str(folder.relative_to(repo)),
                'summary_sha256': hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                'traces_sha256': hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                'tasks': len(by_task), 'runs': len(metrics), 'trace_count': trace_count,
                'stored_strict_success_rate': raw['summary_by_agent']['regex']['strict_success_rate'],
                'recomputed': reliability,
                'all_three_successful_tasks': fully_successful,
                'task_all_three_wilson95_descriptive': wilson(fully_successful, len(by_task)),
                'mixed_outcome_tasks': {t: sum(v) for t, v in by_task.items() if 0 < sum(v) < 3},
                'trace_rescore_authoritative': bool(args.snapshot),
                'trace_rescore_differences': drift,
            })
    comparisons = []
    for model in ['GRPO', 'GSPO', 'DrGRPO', 'DAPO', 'SFT-T5']:
        for split in suffixes:
            old = decisions[('SFT', split)]
            new = decisions[(model, split)]
            if set(old) != set(new):
                raise ValueError(f'Unpaired task pool: {model}: {split}')
            gains = sum(not old[t] and new[t] for t in old)
            losses = sum(old[t] and not new[t] for t in old)
            comparisons.append({'model': model, 'reference': 'SFT', 'split': split,
                                'paired_tasks': len(old), 'gains': gains, 'losses': losses,
                                'delta_pass3': (gains - losses) / len(old)})
    split_files = {}
    for filename in ['v3_81.json', 'v6_grpo_train.json']:
        p = repo / 'data' / 'splits' / filename
        split_files[filename] = {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                                 'content': json.loads(p.read_text())}
    output = {'kind': 'archived_metric_reproduction',
              'fresh_model_inference': False,
              'trace_rescore_note': ('Original snapshot supplied.' if args.snapshot else
                                    'Original snapshot missing: unbound trace rescoring can reject semantically equivalent refs. Differences are not confirmed regressions.'),
              'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
              'evaluations': evaluations, 'paired_comparisons': comparisons,
              'splits': split_files,
              'limitations': ['Archived histories only; no independent model regeneration.',
                              'Wilson intervals describe task outcomes; they do not account for template correlation.',
                              'Three trials per task and one training seed cannot establish algorithm superiority.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print('model split tasks runs stored_strict pass1 pass3 drift')
    for e in evaluations:
        print(e['model'], e['split'], e['tasks'], e['runs'],
              round(e['stored_strict_success_rate'], 4),
              round(e['recomputed']['pass^1'], 4), round(e['recomputed']['pass^3'], 4),
              len(e['trace_rescore_differences']))


if __name__ == '__main__':
    main()
