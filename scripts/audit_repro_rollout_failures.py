#!/usr/bin/env python3
"""Audit complete train-only rollouts with frozen oracles; never alter trajectories."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['traces','oracles','split','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'AgenticArxiv'))
    from benchmark.metrics import extract_metrics,is_strict_success
    oracles={t['id']:t for t in json.loads(a.oracles.read_text())}
    eligible=set(json.loads(a.split.read_text())['split']['probe'])
    rows=[json.loads(x) for x in a.traces.read_text().splitlines()]
    assert {r['task_id'] for r in rows}==eligible
    failed=[];counts=Counter();positive_failures=0
    for r in rows:
        assert r['task_id'] in eligible
        metric=extract_metrics(oracles[r['task_id']],r['trajectory'],'regex',r['generation_index'])
        if is_strict_success(metric):continue
        if r['reward']>0:positive_failures+=1
        labels=[]
        for label,condition in [('argument_mismatch',metric.arg_score<1),('reference_mismatch',metric.ref_score<1),('tool_sequence_mismatch',not metric.tool_call_accurate),('tool_execution_failure',metric.tool_exec_failures>0),('parse_failure',metric.parse_failures>0),('blocked_reason_mismatch',not metric.terminal_semantics_accurate),('false_finish',metric.false_finish),('completion_clipped',r['trajectory']['clipped']),('not_completed',not metric.task_completed)]:
            if condition:labels.append(label);counts[label]+=1
        task=oracles[r['task_id']]
        failed.append({'task_id':r['task_id'],'group_index':r['group_index'],'generation_index':r['generation_index'],'reward':r['reward'],'fault_labels':labels,'user_request':task['task'],'expected_tools':task.get('expected_tools'),'expected_tool_args':task.get('expected_tool_args'),'metrics':asdict(metric),'history':r['trajectory']['history']})
    result={'samples':len(rows),'strict_failures':len(failed),'fault_counts_overlapping':dict(counts),'positive_reward_strict_failures':positive_failures,'full_reward_strict_failures':sum(x['reward']>=1-1e-6 for x in failed),'trace_sha256':hashlib.sha256(a.traces.read_bytes()).hexdigest(),'note':'Fault categories overlap; positive partial credit does not imply strict success. Oracles are used only for audit, not model prompts.','failures':failed}
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='failures'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
