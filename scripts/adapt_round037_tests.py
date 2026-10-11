from pathlib import Path
B=Path('/data/home/chenyizhou/AgenticArXiv-RL')
p=B/'AgenticArxiv/tests/test_task_spec.py';s=p.read_text();old='        runner.llm_extra = None\n        runner._llm_client = object()';new='        runner.llm_extra = None\n        runner.prompt_policy = "baseline"\n        runner._llm_client = object()';assert s.count(old)==1;p.write_text(s.replace(old,new))
p=B/'AgenticArxiv/tests/test_request_contract.py';s=p.read_text();assert 'test_routed_prompt_preserves_explicit_contract_policy' not in s;s+='''

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
''';p.write_text(s)
print('Budget fixture initialized for preserved prompt policy; combined policy/routing behavior test added')
