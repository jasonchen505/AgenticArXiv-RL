#!/usr/bin/env python3
"""Recompute train-only probe quality, grouping and frozen-policy identity."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--source_model',type=Path,required=True)
    p.add_argument('--oracles',type=Path,required=True)
    p.add_argument('--split',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--groups_per_task',type=int,default=2)
    p.add_argument('--num_generations',type=int,default=4)
    a=p.parse_args()
    assert a.groups_per_task>0 and a.num_generations>=2
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'AgenticArxiv'))
    from benchmark.metrics import extract_metrics,is_strict_success
    from safetensors.torch import load_file
    import torch
    manifest=json.loads((a.directory/'final/training_manifest.json').read_text())
    assert manifest['learning_rate']==0 and manifest['frozen_policy_probe']
    rows=[json.loads(line) for line in (a.directory/'rollout_traces.jsonl').read_text().splitlines()]
    oracles={t['id']:t for t in json.loads(a.oracles.read_text())}
    eligible=set(json.loads(a.split.read_text())['split']['probe'])
    tasks,groups=defaultdict(list),defaultdict(list)
    for row in rows:
        assert row['task_id'] in eligible
        trajectory=row['trajectory']
        metric=extract_metrics(oracles[row['task_id']],trajectory,'regex',row['generation_index'])
        good=is_strict_success(metric)
        tasks[row['task_id']].append({'reward':row['reward'],'strict_success':good,'clipped':bool(trajectory['clipped']),'policy_tokens':trajectory['token_usage']['policy_tokens'],'completion_tokens':trajectory['token_usage']['completion_tokens'],'false_finish':metric.false_finish,'argument_mismatch':metric.arg_score<1,'tool_mismatch':not metric.tool_call_accurate,'tool_fail':metric.tool_exec_failures>0,'terminal_mismatch':not metric.terminal_semantics_accurate})
        groups[(row['task_id'],row['group_index'])].append(row)
    assert set(tasks)==eligible
    assert all(len(group)==a.num_generations and {row['generation_index'] for row in group}==set(range(a.num_generations)) for group in groups.values())
    assert all(len(entries)==a.groups_per_task*a.num_generations for entries in tasks.values()),'Per-task coverage mismatch'
    assert manifest['actual_steps']==len(eligible)*a.groups_per_task
    before=load_file(str(a.source_model/'adapter_model.safetensors'))
    after=load_file(str(a.directory/'final/adapter_model.safetensors'))
    unchanged=set(before)==set(after) and all(torch.equal(before[k],after[k]) for k in before)
    assert unchanged,'lr=0 probe altered policy tensors'
    per_task={}
    for task,entries in tasks.items():
        values=[entry['reward'] for entry in entries]
        task_groups=[group for (t,_),group in groups.items() if t==task]
        informative=sum(max(row['reward'] for row in group)-min(row['reward'] for row in group)>1e-6 for group in task_groups)
        success=sum(entry['strict_success'] for entry in entries)
        per_task[task]={'samples':len(entries),'groups':len(task_groups),'informative_groups':informative,'mean_reward':sum(values)/len(values),'strict_success_count':success,'strict_success_rate':success/len(entries),'clipped_count':sum(entry['clipped'] for entry in entries),'false_finish_count':sum(e['false_finish'] for e in entries),'argument_mismatch_count':sum(e['argument_mismatch'] for e in entries),'tool_mismatch_count':sum(e['tool_mismatch'] for e in entries),'tool_failure_count':sum(e['tool_fail'] for e in entries),'terminal_mismatch_count':sum(e['terminal_mismatch'] for e in entries),'min_reward':min(values),'max_reward':max(values),'signal_class':'variable_reward' if informative else 'zero_variance_all_strict' if success==len(entries) else 'zero_variance_not_all_strict'}
    result={'kind':'verified_train_only_frozen_probe','samples':len(rows),'groups':len(groups),'tasks':len(tasks),'adapter_tensors_unchanged':unchanged,'actual_steps':manifest['actual_steps'],'clipped_samples':sum(e['clipped'] for entries in tasks.values() for e in entries),'strict_success_count':sum(t['strict_success_count'] for t in per_task.values()),'informative_groups':sum(t['informative_groups'] for t in per_task.values()),'selected_task_ids':[t for t,v in per_task.items() if v['informative_groups']>=2],'per_task':per_task,'source_adapter_sha256':hashlib.sha256((a.source_model/'adapter_model.safetensors').read_bytes()).hexdigest(),'rollout_sha256':hashlib.sha256((a.directory/'rollout_traces.jsonl').read_bytes()).hexdigest(),'reward_summary_sha256':hashlib.sha256((a.directory/'reward_probe_summary.json').read_bytes()).hexdigest(),'limits':[f'Predeclared {len(eligible)}-task train-only probe; {a.groups_per_task} groups per task, {a.num_generations} samples per group.','Zero variance does not imply ceiling performance.','Any selected subset is probe-selected pending training; no claimed GRPO gains.']}
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='per_task'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
