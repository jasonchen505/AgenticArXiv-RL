#!/usr/bin/env python3
"""Record two resource-driven pool swaps; keep raw world and tool results intact."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    a = p.parse_args()
    if a.input.resolve() == a.output.resolve():
        raise SystemExit('Raw snapshot must be preserved')
    raw = json.loads(a.input.read_text())
    curated = deepcopy(raw)
    changes = []
    for aspect, first, second, reason in [
        ('CV', 4, 5, 'Declared read_cv5_method requires a recorded method section at ref=5'),
        ('RO', 2, 4, 'Declared analyze_ro5_ref2_axes requires at least two extracted raster figures'),
    ]:
        candidates = [e for e in curated['get_recently_submitted_cs_papers'].values()
                      if e['args']['aspect'] == aspect]
        if len(candidates) != 1:
            raise SystemExit(f'Expected one frozen {aspect} pool')
        pool = candidates[0]['result']
        before = [paper['id'] for paper in pool]
        pool[first - 1], pool[second - 1] = pool[second - 1], pool[first - 1]
        after = [paper['id'] for paper in pool]
        assert sorted(before) == sorted(after) and len(set(after)) == len(after)
        changes.append({'aspect': aspect, 'positions': [first, second],
                        'before_ids': before[:5], 'after_ids': after[:5], 'reason': reason})
    cv = next(e['result'] for e in curated['get_recently_submitted_cs_papers'].values() if e['args']['aspect']=='CV')
    method_key = json.dumps({'paper_id': cv[4]['id'], 'section': 'method'}, sort_keys=True)
    assert method_key in curated['get_paper_content']
    ro = next(e['result'] for e in curated['get_recently_submitted_cs_papers'].values() if e['args']['aspect']=='RO')
    figure_counts = {e['result']['paper_id']:e['result']['count'] for e in curated['extract_paper_figures'].values()}
    assert figure_counts[ro[1]['id']] >= 2
    assert cv[:3] == next(e['result'] for e in raw['get_recently_submitted_cs_papers'].values() if e['args']['aspect']=='CV')[:3]
    for name in raw:
        if name != 'get_recently_submitted_cs_papers':
            assert raw[name] == curated[name]
    a.output.write_text(json.dumps(curated, ensure_ascii=False, indent=2) + '\n')
    manifest = {'kind': 'resource_curated_offline_world', 'input': str(a.input),
                'input_sha256': digest(a.input), 'output': str(a.output),
                'output_sha256': digest(a.output), 'changes': changes,
                'task_definitions_modified': False, 'tool_observations_modified': False,
                'model_outputs_used_for_curation': False,
                'note': 'Pool order is curated for declared resource capabilities; this is not live arXiv ranking and not the original author snapshot.'}
    a.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
