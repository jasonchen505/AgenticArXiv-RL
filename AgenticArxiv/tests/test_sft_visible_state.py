"""Prevent SFT from hiding state which benchmark and RL expose to the policy."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from generate_sft_data import build_sft_samples_from_history, generate_deterministic_trajectories
from agents.prompt_templates import build_visible_setup_context
from benchmark.task_spec import Step, TaskSpec


class ProbeEnv:
    def reset_runtime_state(self):
        pass

    def execute_tool(self, name, args):
        if name == 'get_recently_submitted_cs_papers':
            return [{'id': '2601.00001v1', 'title': 'State Probe Paper',
                     'authors': ['Tester'], 'pdf_url': 'https://arxiv.org/pdf/2601.00001v1'}]
        return {'paper_id': '2601.00001v1', 'status': 'READY'}


def test_setup_visible_to_sft_without_becoming_supervised_action():
    spec = TaskSpec(id='visible_state_probe', task='查看刚才下载论文的缓存状态',
                    setup=(Step('get_recently_submitted_cs_papers',
                                {'aspect': 'AI', 'days': 7, 'max_results': 1}),
                           Step('download_arxiv_pdf', {'ref': 1})),
                    steps=(Step('get_paper_cache_status', {'ref': None}),))
    rows = generate_deterministic_trajectories([spec], ProbeEnv(), 'tools')
    assert len(rows) == 2  # cache and FINISH; setup is not a supervised target
    state = build_visible_setup_context(spec.to_task())
    for row in rows:
        assert state in row['messages'][0]['content']
        assert 'download_arxiv_pdf' not in row['messages'][1]['content']
        assert 'expected_tools' not in row['messages'][0]['content']


def test_same_request_has_distinct_observation_for_empty_and_active_sessions():
    history = [{'thought': '完成', 'action': 'FINISH', 'observation': 'done'}]
    task = '处理刚才那篇论文'
    empty = build_visible_setup_context({})
    active = build_visible_setup_context({'setup': [
        {'name': 'download_arxiv_pdf', 'args': {'ref': 2}}]})
    empty_rows = build_sft_samples_from_history(task, 'tools', history, initial_history=empty)
    active_rows = build_sft_samples_from_history(task, 'tools', history, initial_history=active)
    assert empty_rows[0]['messages'][0]['content'] != active_rows[0]['messages'][0]['content']
    assert '没有既有论文列表' in empty_rows[0]['messages'][0]['content']
    assert '最近操作的论文' in active_rows[0]['messages'][0]['content']


def test_legacy_dependency_search_is_described_as_existing_state():
    spec = TaskSpec(id='dependency_probe', task='下载第一篇论文', depends_on='search_01',
                    steps=(Step('download_arxiv_pdf', {'ref': 1}),))
    rows = generate_deterministic_trajectories([spec], ProbeEnv(), 'tools')
    prompt = rows[0]['messages'][0]['content']
    assert '此前已检索' in prompt
    assert '没有既有论文列表' not in prompt
