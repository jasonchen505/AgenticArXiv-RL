#!/usr/bin/env python3
"""Measure compact gold policy + native observation tokens, without GPU sampling.

This is a task-specific lower-cost reference, not an upper bound on real policy
generation. It mirrors the frozen rollout's 1000-character observation limit.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--split', required=True)
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'AgenticArxiv'))
    os.environ['STORE_BACKEND'] = 'memory'
    with tempfile.TemporaryDirectory(prefix='arxiv-budget-') as directory:
        for var in ['PDF_RAW_PATH','PDF_TRANSLATED_PATH','PDF_FIGURES_PATH','PDF_CACHE_PATH','TRANSLATE_CACHE_PATH','PDF_TRANSLATED_LOG_PATH']:
            os.environ[var] = str(Path(directory)/var.lower())
        from transformers import AutoTokenizer
        from benchmark.task_io import load_tasks
        from benchmark.semantic_oracle import attach_expected_paper_ids
        from benchmark.task_spec import reference_terminal_thought
        from rl.grpo_reward import _prepare_rollout_environment, _dispatch_environment_tool
        from rl.multiturn_env import AgenticArxivMultiTurnEnv
        from rl.reward import RewardCalculator
        from benchmark.metrics import is_strict_success
        from tools.bootstrap import require_all_tools
        require_all_tools('Gold completion budget audit')
        tokenizer = AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
        reward = RewardCalculator(curriculum_steps=0)
        tasks = attach_expected_paper_ids(load_tasks('expanded',a.split), a.snapshot)
        rows = []
        for task in tasks:
            environment = AgenticArxivMultiTurnEnv(a.snapshot)
            _prepare_rollout_environment(environment,task['id'],{task['id']:task})
            policy, observations, steps, history = 0,0,[],[]
            for name,args in zip(task['expected_tools'], task['expected_tool_args']):
                text = 'Thought: 执行任务要求\nAction: '+json.dumps({'name':name,'args':args},ensure_ascii=False)
                full_observation = str(_dispatch_environment_tool(environment,name,args or {}))
                observation = full_observation[:1000]
                suffix = f'\nObservation: {observation}\nThought:'
                policy_tokens = len(tokenizer(text,add_special_tokens=False)['input_ids'])
                env_tokens = len(tokenizer(suffix,add_special_tokens=False)['input_ids'])
                policy+=policy_tokens; observations+=env_tokens
                steps.append({'tool':name,'policy_tokens':policy_tokens,'observation_tokens':env_tokens,'observation_chars':len(observation)})
                history.append({'thought':'执行任务要求','action':json.dumps({'name':name,'args':args},ensure_ascii=False),'observation':full_observation,'model_observation':observation})
            thought = reference_terminal_thought(task) if task.get('expected_terminal_mode')=='blocked' else '任务完成'
            finish = len(tokenizer(f'Thought: {thought}\nAction: FINISH',add_special_tokens=False)['input_ids'])
            policy += finish
            history.append({'thought':thought,'action':'FINISH','observation':'任务完成'})
            breakdown,metrics = reward.compute_reward_breakdown(task,{'history':history})
            rows.append({'task_id':task['id'],'turns_required':len(steps)+1,'compact_policy_tokens':policy,'native_observation_tokens':observations,'compact_total_tokens':policy+observations,'finish_policy_tokens':finish,'steps':steps,'native_gold_strict_success':is_strict_success(metrics),'native_gold_reward':breakdown.total})
            environment.reset()
        result = {'kind':'compact_gold_completion_budget','fresh_model_inference':False,'split':a.split,'snapshot_sha256':hashlib.sha256(a.snapshot.read_bytes()).hexdigest(),'observation_char_limit':1000,'native_gold_strict_passed':sum(r['native_gold_strict_success'] for r in rows),'native_gold_count':len(rows),'native_gold_min_reward':min(r['native_gold_reward'] for r in rows),'max_turns_required':max(row['turns_required'] for row in rows),'max_compact_total_tokens':max(row['compact_total_tokens'] for row in rows),'reference_exceeds_budget':{str(b):[row['task_id'] for row in rows if row['compact_total_tokens']>b] for b in [256,1024,2048,4096]},'tasks':rows,'limits':['Compact reference, not a proof that all sampled completions fit.','Native observation suffixes consume completion budget although they are excluded from policy loss.','Per-turn EOS and sampler-specific boundary token costs are not counted; UUID paths can change small token counts.','A zero-variance group can be floor or ceiling; classify using quality, not variance alone.']}
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='tasks'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
