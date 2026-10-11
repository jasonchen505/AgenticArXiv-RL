#!/usr/bin/env python3
"""Paired comparison with authoritative rescoring against frozen task oracles."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--sft', type=Path, required=True)
    p.add_argument('--oracles', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sft_label', default='SFT30')
    p.add_argument('--base_label', default='Base')
    p.add_argument('--training_description', default='one 30-step training run')
    p.add_argument('--evaluation_split', choices=['dev','iid_test','ood_test'], default='dev')
    a = p.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'AgenticArxiv'))
    from benchmark.metrics import extract_metrics, is_strict_success
    from benchmark.report import BenchmarkReport
    oracles = {t['id']: t for t in json.loads(a.oracles.read_text())}
    decisions, results = {}, {}
    for name, directory in [(a.base_label, a.base), (a.sft_label, a.sft)]:
        summary = json.loads((directory/'summary.json').read_text())
        if summary['error_count']:
            raise SystemExit('Evaluation contains exceptions; refusing partial comparison')
        records = [json.loads(line) for line in (directory/'traces.jsonl').read_text().splitlines()]
        raw_details = {(d['task_id'],d['trial']):d for d in summary['details']}
        metrics, per_task = [], defaultdict(list)
        seen = set()
        for trace in records:
            key = (trace['task_id'],trace['trial'])
            if key in seen:
                raise SystemExit('Duplicate evaluation trial')
            seen.add(key)
            metric = extract_metrics(oracles[key[0]], trace, 'regex', key[1])
            good = is_strict_success(metric)
            d = raw_details[key]
            stored = (d['completed'] and d['tool_accurate'] and d['arg_score']==1
                      and d['ref_score']==1 and d['parse_fail']==0 and d['tool_fail']==0
                      and d['terminal_semantics_accurate'])
            if bool(good) != bool(stored):
                raise SystemExit(f'Authoritative metric drift: {name} {key}')
            metrics.append(metric)
            per_task[key[0]].append(good)
        if seen != set(raw_details) or any(len(v)!=3 for v in per_task.values()):
            raise SystemExit('Incomplete task/trial coverage')
        decisions[name] = {key:all(values) for key,values in per_task.items()}
        results[name] = {'runs':len(metrics),'tasks':len(per_task),
                         'strict_success_rate':summary['summary_by_agent']['regex']['strict_success_rate'],
                         'reliability':BenchmarkReport(metrics).reliability_by_agent()['regex'],
                         'false_finish_rate':summary['summary_by_agent']['regex']['false_finish_rate'],
                         'avg_tokens':summary['summary_by_agent']['regex']['avg_tokens'],
                         'authoritative_rescore_drift':0,
                         'traces_sha256':hashlib.sha256((directory/'traces.jsonl').read_bytes()).hexdigest()}
    before, after = decisions[a.base_label], decisions[a.sft_label]
    if set(before)!=set(after):
        raise SystemExit('Unpaired task pools')
    gains = sum(not before[t] and after[t] for t in before)
    losses = sum(before[t] and not after[t] for t in before)
    n = gains+losses
    exact_p = min(1,2*sum(math.comb(n,k) for k in range(min(gains,losses)+1))/2**n) if n else 1
    result = {'kind':('paired_heldout_comparison' if a.evaluation_split!='dev' else 'paired_dev_smoke_comparison' if a.sft_label=='SFT30' else 'paired_dev_comparison'),'evaluation_split':a.evaluation_split,'models':results,
              'task_gains':gains,'task_losses':losses,
              'paired_task_exact_two_sided_p_descriptive':exact_p,
              'task_oracles_sha256':hashlib.sha256(a.oracles.read_bytes()).hexdigest(),
              'limitations':[f'{a.evaluation_split}, {len(before)} tasks, 3 trials/task, {a.training_description}.',
                             'New resource-curated caption world; not original author metrics.',
                             'No controlled ablation of the setup visibility fix; this report covers only the named split.']}
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
