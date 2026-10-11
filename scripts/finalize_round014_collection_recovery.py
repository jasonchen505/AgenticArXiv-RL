#!/usr/bin/env python3
"""Certify frozen-policy collection and inventory failures without auto-training."""
from pathlib import Path
import collections,hashlib,json,shutil,sys
R=Path('artifacts/reproduction/round-013-2026-10-05');OUT=Path('outputs/repro_round013/collection')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
    plan=read(R/'collection-plan.json')
    for p,h in plan['sha256'].items():assert sha(p)==h,p
    job=read(R/'collection-job.json');assert job['exit_code']==0 and job['gpu_index']==4
    manifest=read(OUT/'final/training_manifest.json');assert manifest['actual_steps']==274 and manifest['learning_rate']==0. and manifest['frozen_policy_probe']
    from safetensors.torch import load_file
    import torch
    source=load_file(plan['source_model']+'/adapter_model.safetensors');end=load_file(str(OUT/'final/adapter_model.safetensors'));ref=load_file(str(OUT/'final/ref/adapter_model.safetensors'))
    assert len(source)==392 and set(source)==set(end)==set(ref)
    assert all(torch.equal(source[k],end[k]) and torch.equal(source[k],ref[k]) for k in source)
    rows=[json.loads(s) for s in (OUT/'rollout_traces.jsonl').read_text().splitlines()]
    tasks={t['id']:t for t in read(R/'collection-oracles.json')};counts=collections.Counter(x['task_id'] for x in rows)
    assert len(rows)==1096 and set(counts)==set(tasks) and set(counts.values())=={8}
    sys.path.insert(0,'AgenticArxiv');from benchmark.metrics import extract_metrics,is_strict_success
    failed=[];success=0;parents=set();clipped=0
    for i,x in enumerate(rows):
        task=tasks[x['task_id']];parents.add(task['parent_task_id']);tr=x['trajectory'];m=extract_metrics(task,tr,agent_type="frozen_sft_native",trial=i);ok=is_strict_success(m);success+=int(ok);clipped+=int(tr.get('clipped',False))
        if not ok:failed.append({'row_index':i,'task_id':x['task_id'],'parent_task_id':task['parent_task_id'],'group_index':x['group_index'],'generation_index':x['generation_index'],'reward':x['reward'],'metrics':m.to_dict(),'raw_assistant_turns':x['raw_assistant_turns'],'trajectory':tr,'admitted_to_training':False})
    write(R/'failure-inventory.json',{'model_rollouts':1096,'strict_successes':success,'strict_failures':len(failed),'semantic_coverage':137,'train_parent_coverage':len(parents),'clipped':clipped,'failure_semantic_support':len({x['task_id'] for x in failed}),'failure_parent_support':len({x['parent_task_id'] for x in failed}),'counts_by_task':dict(collections.Counter(x['task_id'] for x in failed)),'failure_is_not_automatically_valid_negative':True,'negative_training_dataset_created':False,'admission_policy':plan['negative_admission_policy'],'failures':failed})
    for f in ['rollout_traces.jsonl','rollout_traces.summary.json','reward_probe_summary.json']:shutil.copy2(OUT/f,R/f)
    shutil.copy2(OUT/'final/training_manifest.json',R/'collection-training-manifest.json')
    write(R/'collection-completion.json',{'state':'completed','actual_model_rollouts':1096,'semantic_coverage':137,'train_parent_coverage':len(parents),'heldout_parent_overlap':0,'samples_per_semantic':8,'native_lr0_loop_steps':274,'learning_updates':0,'source_policy_tensor_equal':392,'reference_tensor_equal':392,'strict_successes':success,'strict_failures':len(failed),'clipped':clipped,'process_seconds':job['elapsed_seconds'],'new_model_improvement_claimed':False,'next_step':'Validate first-divergence or certified-recovery preference pairs and negative support, then match against chosen-only SFT; no automatic training on raw failures.'})
    print(f'Frozen collection complete:1096rollouts,137semantics,strictfailures{len(failed)},392unchangedsource/ref tensors',flush=True)
if __name__=='__main__':main()
