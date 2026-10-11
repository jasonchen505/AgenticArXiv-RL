#!/usr/bin/env python3
"""Build a new, frozen train-only dataset; never reuse old row-count assumptions."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--split', type=Path, required=True)
    p.add_argument('--gold-report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    gold = json.loads(args.gold_report.read_text())
    if not gold['training_gate_passed'] or gold['snapshot_sha256'] != sha(args.snapshot):
        raise SystemExit('Gold coverage gate failed or snapshot changed; refusing training data')
    repo = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repo / 'scripts'), str(repo / 'AgenticArxiv')]
    os.environ['STORE_BACKEND'] = 'memory'
    with tempfile.TemporaryDirectory(prefix='agenticarxiv-data-') as scratch:
        for var in ['PDF_RAW_PATH', 'PDF_TRANSLATED_PATH', 'PDF_TRANSLATED_LOG_PATH',
                    'PDF_CACHE_PATH', 'TRANSLATE_CACHE_PATH', 'PDF_FIGURES_PATH']:
            os.environ[var] = str(Path(scratch) / var.lower())
        from generate_sft_data import select_task_specs, generate_deterministic_trajectories
        from generate_parametric_sft_data import build_parametric_tasks, validate_derived_tasks
        from augment_sft_data import TASK_WRAPPERS, replace_prompt_task, split_prompt_task, decision_thought, split_assistant_action
        from agents.prompt_templates import format_tool_description
        from benchmark.tasks_expanded import get_expanded_tasks
        from rl.env import MockArxivEnv
        from tools.bootstrap import require_all_tools
        from tools.tool_registry import registry
        require_all_tools('Reproduction SFT data')
        split_payload = json.loads(args.split.read_text())
        original, lineage = select_task_specs('expanded', f'{args.split}:train')
        derived = build_parametric_tasks(include_t5=True)
        validate_derived_tasks(derived, split_payload)
        specs = original + [d.spec for d in derived]
        parents = {s.id: s.id for s in original}
        parents.update({d.spec.id: d.parent_task_id for d in derived})
        contracts = {s.id: s.to_task() for s in specs}
        train = set(split_payload['split']['train'])
        heldout = set(sum((split_payload['split'][name] for name in ['dev', 'iid_test', 'ood_test']), []))
        if set(parents.values()) - train or set(parents.values()) & heldout:
            raise SystemExit('Parent task leakage')
        heldout_texts = {t['task'].strip() for t in get_expanded_tasks() if t['id'] in heldout}
        env = MockArxivEnv(snapshot_path=args.snapshot, mode='replay')
        desc = format_tool_description(registry.list_tools())
        seed = generate_deterministic_trajectories(specs, env, desc, source_split=lineage)
        rows, seen = [], set()
        for sample in seed:
            prompt, assistant = [m['content'] for m in sample['messages']]
            _, task, _ = split_prompt_task(prompt)
            thought, action = split_assistant_action(assistant)
            for task_variant, wrapper in enumerate(TASK_WRAPPERS):
                text = wrapper.format(task=task)
                if text.strip() in heldout_texts:
                    raise SystemExit('Heldout request text leaked')
                varied_prompt = replace_prompt_task(prompt, text)
                for thought_variant, current_thought in enumerate(
                    [thought, decision_thought(action, contracts[sample['source_task_id']])]
                ):
                    messages = [{'role': 'user', 'content': varied_prompt},
                                {'role': 'assistant', 'content': f'{current_thought}\nAction:{action}'}]
                    fingerprint = hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                    if fingerprint in seen:
                        raise SystemExit('Duplicate augmented sample')
                    seen.add(fingerprint)
                    rows.append({**sample, 'messages': messages,
                                 'parent_task_id': parents[sample['source_task_id']],
                                 'sample_sha256': fingerprint,
                                 'augmentation': {'task_variant': task_variant, 'thought_variant': thought_variant}})
        random.Random(42).shuffle(rows)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        manifest = {'kind': 'qlora_sft_train_mix', 'protocol': 'reproduction_20261002_v1',
                    'output_sha256': sha(args.output), 'output_rows': len(rows),
                    'unique_sample_fingerprints': len(seen), 'semantic_task_instances': len(specs),
                    'original_train_tasks': len(original), 'parametric_tasks': len(derived),
                    'seed_rows': len(seed), 'heldout_parent_overlap': 0,
                    'snapshot_sha256': sha(args.snapshot), 'split_sha256': sha(args.split),
                    'gold_report_sha256': sha(args.gold_report), 'source_split': lineage,
                    'setup_visible': True, 'figure_backend': 'extractive_caption', 'shuffle_seed': 42,
                    'rows_by_parent': dict(Counter(r['parent_task_id'] for r in rows)),
                    'note': 'New protocol. Linguistic variants are not independent task semantics.'}
        args.output.with_suffix(args.output.suffix + '.manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({k: v for k, v in manifest.items() if k != 'rows_by_parent'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
