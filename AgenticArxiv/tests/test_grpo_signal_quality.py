"""Quality regressions for variance-based task selection; CPU only."""
import json
import sys
from pathlib import Path
from scripts.build_grpo_signal_split import select_tasks, build_document


def entry(reward, informative=0):
    return {"sample_count":8,"group_count":2,"mean_reward":reward,"min_reward":reward,"max_reward":reward,"informative_group_fraction":informative/2}


def test_uniform_failure_is_not_ceiling():
    b=select_tasks({"bad":entry(-0.75),"good":entry(1)},["bad","good"],2)
    assert b["ceiling_control"]==["good"]
    assert b["zero_variance_low_quality"]==["bad"]


def test_uniform_partial_credit_is_not_ceiling():
    b=select_tasks({"partial":entry(0.5)},["partial"],2)
    assert b["ceiling_control"]==[]
    assert b["zero_variance_low_quality"]==["partial"]


def test_missing_reward_extrema_cannot_certify_ceiling():
    t=entry(1);t.pop("min_reward")
    b=select_tasks({"unknown":t},["unknown"],2)
    assert b["ceiling_control"]==[]
    assert b["zero_variance_low_quality"]==["unknown"]


def test_strict_oracle_can_veto_full_reward_ceiling():
    b=select_tasks({"false_positive":entry(1)},["false_positive"],2,quality_tasks={"false_positive":{"samples":8,"strict_success_count":7}})
    assert b["ceiling_control"]==[]
    assert b["zero_variance_low_quality"]==["false_positive"]


def test_document_preserves_failure_bucket_and_eligibility(tmp_path):
    tasks={"signal":entry(0.3,2),"ceiling":entry(1),"floor":entry(-0.75),"partial":entry(0.5,1)}
    probe={"tasks":tasks};p=tmp_path/"probe.json";p.write_text(json.dumps(probe))
    buckets=select_tasks(tasks,list(tasks),2)
    doc=build_document(probe_path=p,probe=probe,source_ref="s:train",model="m",num_generations=4,seed=43,min_informative_groups=2,buckets=buckets)
    assert doc["split"]["zero_variance_low_quality"]==["floor"]
    assert doc["summary"]["eligible_task_count"]==4
    assert doc["summary"]["zero_variance_low_quality_task_count"]==1
    assert "floor" in doc["audit"]


import hashlib
import pytest
from scripts import build_grpo_signal_split as selector


@pytest.mark.parametrize('mismatch',['reward','adapter'])
def test_cli_rejects_stale_quality_before_output(tmp_path,monkeypatch,mismatch):
    probe=tmp_path/'probe.json';probe.write_text(json.dumps({'tasks':{'signal':entry(0.3,2)}}))
    split=tmp_path/'split.json';split.write_text(json.dumps({'split':{'train':['signal']}}))
    model=tmp_path/'model';model.mkdir();adapter=model/'adapter_model.safetensors';adapter.write_bytes(b'CPU fixture: no model load')
    quality={'adapter_tensors_unchanged':True,'reward_summary_sha256':hashlib.sha256(probe.read_bytes()).hexdigest(),'source_adapter_sha256':hashlib.sha256(adapter.read_bytes()).hexdigest(),'per_task':{}}
    quality['reward_summary_sha256' if mismatch=='reward' else 'source_adapter_sha256']='stale_hash'
    q=tmp_path/'quality.json';q.write_text(json.dumps(quality))
    output=tmp_path/'result.json'
    monkeypatch.setattr(sys,'argv',['selector','--probe',str(probe),'--source-split',str(split)+':train','--model',str(model),'--output',str(output),'--quality-summary',str(q)])
    with pytest.raises(SystemExit,match='different reward probe' if mismatch=='reward' else 'different policy adapter'):
        selector.main()
    assert not output.exists()
