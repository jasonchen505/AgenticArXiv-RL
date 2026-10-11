"""Integration checks for opt-in prompts; no model or network."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agents.agent_engine import ReActAgent
from agents.prompt_templates import REACT_PROMPT_TEMPLATE, REQUEST_CONTRACT_V1
from benchmark.runner import BenchmarkRunner


def test_default_agent_retains_exact_baseline_prompt():
    agent=ReActAgent(object())
    task='读取论文 {"id":"x"}，保留用户原文'
    history='Thought: 失败\nAction: FINISH\nObservation: missing'
    messages,payload=agent.build_messages(task,'tool schema',history)
    assert messages==[{"role":"user","content":REACT_PROMPT_TEMPLATE.format(task=task,tools_description='tool schema',history=history)}]
    assert payload=={}


def test_contract_is_opt_in_and_keeps_raw_request_history():
    baseline=ReActAgent(object())
    candidate=ReActAgent(object(),prompt_policy='contract_v1')
    task='关键词 "保留引号" {x}，max_results=7；现在开始执行任务：\n保留这段原文'
    history='Observation: 未执行，请勿称已完成'
    normal,_=baseline.build_messages(task,'tool schema',history)
    improved,payload=candidate.build_messages(task,'tool schema',history)
    assert REQUEST_CONTRACT_V1 not in normal[0]['content']
    assert REQUEST_CONTRACT_V1 in improved[0]['content']
    assert improved[0]['content'].endswith(history+'\n')
    assert task in improved[0]['content']
    assert payload=={}


def test_runner_rejects_unsupported_policy_before_loading_model():
    with pytest.raises(ValueError,match='regex agent only'):
        BenchmarkRunner(agent_types=['mcp'],prompt_policy='contract_v1')
    with pytest.raises(ValueError,match='Unknown prompt policy'):
        BenchmarkRunner(agent_types=['regex'],prompt_policy='unknown')


def test_routed_prompt_preserves_explicit_contract_policy():
    """The optional routing branch must retain the explicitly selected policy."""
    task='下载 arXiv:2608.14528v1，force=false'
    baseline=ReActAgent(object(),tool_router=None)
    contract=ReActAgent(object(),prompt_policy='contract_v1',tool_router=None)
    plain,_=baseline.build_routed_messages(task,'download_arxiv_pdf','download schema','visible history')
    routed,_=contract.build_routed_messages(task,'download_arxiv_pdf','download schema','visible history')
    assert REQUEST_CONTRACT_V1 not in plain[0]['content']
    assert REQUEST_CONTRACT_V1 in routed[0]['content']
    assert task in routed[0]['content'] and 'visible history' in routed[0]['content']
    assert baseline.tool_router is None and contract.tool_router is None
