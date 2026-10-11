"""Guard against misreporting single-run success as three-trial reliability."""
import json
import tempfile
from pathlib import Path

import pytest
from benchmark.metrics import TaskMetrics
from benchmark.report import BenchmarkReport


def metric(task_id, trial, ok):
    return TaskMetrics(task_id=task_id, agent_type='regex', trial=trial,
                       task_completed=True, tool_call_accurate=ok,
                       termination_type='FINISH')


def export(metrics):
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / 'summary.json'
        BenchmarkReport(metrics).to_json(str(output))
        return json.loads(output.read_text())


def test_json_distinguishes_success_rate_from_all_three_reliability():
    metrics = [metric(task, trial, ok)
               for task, outcomes in {'stable': [True, True, True],
                                      'unstable': [True, True, False]}.items()
               for trial, ok in enumerate(outcomes)]
    data = export(metrics)
    assert data['summary_by_agent']['regex']['strict_success_rate'] == pytest.approx(5 / 6)
    reliability = data['reliability_by_agent']['regex']
    assert reliability['pass^1'] == pytest.approx(5 / 6)
    assert reliability['pass^2'] == pytest.approx(2 / 3)
    assert reliability['pass^3'] == 0.5
    assert reliability['tasks'] == 2
    assert reliability['criterion'] == 'strict'


def test_json_preserves_missing_reliability_and_strict_argument_failure():
    wrong_args = metric('bad_args', 0, True)
    wrong_args.arg_score = 0.5
    data = export([wrong_args, metric('short', 0, True), metric('short', 1, True)])
    reliability = data['reliability_by_agent']['regex']
    assert reliability['pass^1'] == 0.5  # task mean differs from pooled 2/3
    assert reliability['pass^3'] is None
    assert reliability['pass^3_skipped'] == 2
    assert data['summary_by_agent']['regex']['strict_success_rate'] == pytest.approx(2 / 3)
