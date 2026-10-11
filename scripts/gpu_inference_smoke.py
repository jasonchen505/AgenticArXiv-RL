#!/usr/bin/env python3
"""First-action generation smoke check, without tool execution or benchmark claims."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / 'AgenticArxiv'))
    os.environ.setdefault('STORE_BACKEND', 'memory')
    import torch
    from tools.bootstrap import require_all_tools
    from utils.llm_client import TransformersLLMClient
    from agents.agent_engine import ReActAgent
    from agents.prompt_templates import build_visible_setup_context, format_tool_description
    from benchmark.tasks_expanded import get_expanded_tasks
    require_all_tools('GPU inference smoke')
    client = TransformersLLMClient(args.model, device='cuda', dtype='bfloat16', seed=45)
    agent = ReActAgent(client)
    desc = format_tool_description(agent.discover_tools())
    pool = {t['id']: t for t in get_expanded_tasks()}
    ids = ['search_AI_1d_3', 'state_ref_last_active', 'infeasible_unsupported_action']
    results = []
    for tid in ids:
        task = pool[tid]
        messages, _ = agent.build_messages(task['task'], desc, build_visible_setup_context(task))
        client.start_generation_stream(f'{tid}:0')
        torch.cuda.synchronize()
        start = time.perf_counter()
        response = client.chat_completions(args.model, messages, temperature=0.1,
                                          max_tokens=256, extra={'stop': ['Observation:']})
        torch.cuda.synchronize()
        thought, action = agent.parse_response(response)
        content = response['choices'][0]['message']['content']
        # Preserve raw output; parser's None also covers parse failure, so it is
        # never counted as a successful blocked action automatically.
        results.append({'task_id': tid, 'task': task['task'],
                        'prompt_sha256': hashlib.sha256(messages[0]['content'].encode()).hexdigest(),
                        'output': content, 'parsed_thought': thought, 'parsed_action': action,
                        'usage': response['usage'], 'latency_seconds': time.perf_counter() - start})
        print(tid, json.dumps(action, ensure_ascii=False), response['usage'], flush=True)
    data = {'kind': 'fresh_gpu_first_action_smoke', 'model': args.model,
            'gpu_visibility': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'gpu_name': torch.cuda.get_device_name(0), 'temperature': 0.1, 'seed': 45,
            'max_new_tokens': 256, 'tool_execution': False, 'benchmark_success_rate': None,
            'peak_memory_allocated_bytes': torch.cuda.max_memory_allocated(), 'results': results}
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
