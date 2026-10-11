#!/usr/bin/env python3
"""Validate gold tool paths and freeze their semantic targets, without an LLM."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / 'AgenticArxiv'))
    os.environ['STORE_BACKEND'] = 'memory'
    # RolloutSandbox must never own the directories containing frozen inputs.
    with tempfile.TemporaryDirectory(prefix='agenticarxiv-gold-') as scratch:
        for var in ['PDF_RAW_PATH', 'PDF_TRANSLATED_PATH', 'PDF_TRANSLATED_LOG_PATH',
                    'PDF_CACHE_PATH', 'TRANSLATE_CACHE_PATH', 'PDF_FIGURES_PATH']:
            os.environ[var] = str(Path(scratch) / var.lower())
        from benchmark.semantic_oracle import attach_expected_paper_ids, _call
        from benchmark.tasks_expanded import get_expanded_tasks
        from benchmark.task_spec import reference_terminal_thought
        from benchmark.metrics import is_strict_success
        from rl.multiturn_env import AgenticArxivMultiTurnEnv
        from rl.reward import RewardCalculator
        from tools.bootstrap import require_all_tools
        require_all_tools('Frozen world validation')
        reward = RewardCalculator(curriculum_steps=0)
        results, enriched, traces = [], [], []
        for original in get_expanded_tasks():
            try:
                task = attach_expected_paper_ids([original], args.snapshot)[0]
                environment = AgenticArxivMultiTurnEnv(args.snapshot)
                environment.reset(task_id=task['id'])
                for setup in task.get('setup') or []:
                    _call(environment, setup['name'], setup.get('args') or {})
                history = []
                for name, params in zip(task['expected_tools'], task['expected_tool_args']):
                    observation = _call(environment, name, params or {})
                    if isinstance(observation, list) and any(
                        p.get('_mock_env', {}).get('offline_fallback') for p in observation
                        if isinstance(p, dict)
                    ):
                        raise ValueError('Search used an unrecorded fallback pool')
                    history.append({'thought': '执行声明式标准步骤',
                                    'action': json.dumps({'name': name, 'args': params}, ensure_ascii=False),
                                    'observation': json.dumps(observation, ensure_ascii=False, default=str)})
                terminal = (reference_terminal_thought(task)
                            if task.get('expected_terminal_mode') == 'blocked' else '任务已完成')
                history.append({'thought': terminal, 'action': 'FINISH', 'observation': '任务完成'})
                breakdown, metrics = reward.compute_reward_breakdown(task, {'history': history})
                strict = is_strict_success(metrics)
                passed = strict and breakdown.total >= 0.99
                results.append({'task_id': task['id'], 'category': task['category'],
                                'passed': passed, 'strict_success': strict,
                                'reward': breakdown.total, 'reward_components': asdict(breakdown),
                                'metrics': metrics.to_dict()})
                enriched.append(task)
                traces.append({'task_id': task['id'], 'history': history,
                               'expected_paper_ids': task.get('expected_paper_ids')})
                print(task['id'], 'PASS' if passed else 'FAIL', f'reward={breakdown.total:.3f}', flush=True)
            except Exception as exc:
                results.append({'task_id': original['id'], 'category': original['category'],
                                'passed': False, 'error': f'{type(exc).__name__}: {exc}'})
                print(original['id'], 'FAIL', str(exc), flush=True)
        snapshot = json.loads(args.snapshot.read_text())
        counts = {k: len(v) for k, v in snapshot.items() if isinstance(v, dict)}
        failures = [r for r in results if not r['passed']]
        report = {'kind': 'gold_path_validation', 'snapshot': str(args.snapshot),
                  'snapshot_sha256': hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
                  'snapshot_record_counts': counts, 'tasks': len(results),
                  'passed': len(results) - len(failures), 'failed': len(failures),
                  'training_gate_passed': not failures, 'results': results,
                  'figure_backend': 'extractive_caption',
                  'fresh_model_inference': False}
        (args.output / 'gold-validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        (args.output / 'task-oracles.json').write_text(json.dumps(enriched, ensure_ascii=False, indent=2) + '\n')
        (args.output / 'gold-traces.jsonl').write_text(''.join(json.dumps(t, ensure_ascii=False) + '\n' for t in traces))
        print(f'GOLD_GATE: {report["passed"]}/{report["tasks"]}', flush=True)
        if failures:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
