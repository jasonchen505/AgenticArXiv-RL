#!/usr/bin/env python3
"""Finalize only completed equal-budget runs; never infer gains from reward."""
from pathlib import Path
import collections
import hashlib
import json
import subprocess

R=Path('artifacts/reproduction/round-006-2026-10-03')
ARMS=['hard7','broad51']
SPLITS=['dev','iid_test','ood_test']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def good(d):return bool(d['completed'] and d['tool_accurate'] and d['arg_score']==1 and d['ref_score']==1 and d['parse_fail']==0 and d['tool_fail']==0 and d['terminal_semantics_accurate'])

def main():
    completion=read(R/'completion.json');assert completion['state']=='completed'
    plan=read(R/'experiment-plan.json')
    for p,h in plan['sha256'].items():assert sha(p)==h,p
    splits=read('data/splits/v3_81.json')['split']
    training={}
    for a in ARMS:
        frozen=read(R/(a+'-candidate-freeze.json'));m=read(R/(a+'-training-manifest.json'));job=read(R/(a+'-train-job.json'))
        assert sha(Path(frozen['model'])/'adapter_model.safetensors')==frozen['candidate_sha256']
        assert m['actual_steps']==51 and job['exit_code']==0 and not m['reward_variance_guard']['enabled']
        samples=[json.loads(s) for s in (R/(a+'-rollout-traces.jsonl')).read_text().splitlines()]
        groups={(s['batch_index'],s['group_index']) for s in samples}
        assert len(samples)==204 and len(groups)==51
        ids=collections.Counter(s['task_id'] for s in samples)
        assert set(ids)<=set(plan['training_tasks'][a]) and all(v%4==0 for v in ids.values())
        per_group={key:[] for key in groups}
        for s in samples:per_group[(s['batch_index'],s['group_index'])].append(s)
        assert all(len(v)==4 for v in per_group.values())
        informative=sum(v[0]['group_reward_std']>0 for v in per_group.values())
        blocked=sum(n for t,n in ids.items() if t.startswith('infeasible_'))
        training[a]={'optimizer_steps':51,'rollouts':204,'groups':51,'actual_task_coverage':len(ids),'pool_size':len(plan['training_tasks'][a]),'informative_groups':informative,'clipped_rollouts':sum(bool(s['anomalies']['clipped']) for s in samples),'blocked_rollouts':blocked,'blocked_fraction':blocked/204,'task_rollout_counts':dict(sorted(ids.items())),'process_seconds':job['elapsed_seconds'],'torch_allocated_peak_gib':m['peak_vram_gib'],'sampled_device_max_mib':job['sampled_max_device_used_mib'],'train_loss':m['train_loss'],'stop_reason':m['stop_reason'],'candidate_sha256':frozen['candidate_sha256'],'changed_tensors':frozen['changed_tensors']}
    write(R/'training-audit.json',training)
    metrics={m:{} for m in ['sft']+ARMS};total=0;task_counts={};behavior={}
    for split in SPLITS:
        for model in metrics:
            summary=read(R/(model+'-'+split)/'summary.json')
            assert summary['error_count']==0 and summary['sample_count']==3*len(splits[split])
            assert {d['task_id'] for d in summary['details']}==set(splits[split])
            assert len({(d['task_id'],d['trial']) for d in summary['details']})==summary['sample_count']
            assert all(d['trial'] in [0,1,2] for d in summary['details'])
            if model=='sft':
                for f in ['summary.json','traces.jsonl','execution_protocol.json']:assert sha(R/(model+'-'+split)/f)==plan['baseline_hashes'][f'{split}/{f}']
                c=read(R/('sft-vs-hard7-'+split+'.json'))
            else:
                assert read(R/(model+'-'+split+'-job.json'))['exit_code']==0
                c=read(R/('sft-vs-'+model+'-'+split+'.json'));total+=summary['sample_count']
            x=c['models'][model]
            assert x['authoritative_rescore_drift']==0 and c['evaluation_split']==split
            count=sum(good(d) for d in summary['details'])
            assert count==round(x['strict_success_rate']*x['runs'])
            metrics[model][split]={**x,'strict_successes':count,'pass3_tasks':round(x['reliability']['pass^3']*x['tasks'])}
            counts=collections.Counter()
            for d in summary['details']:counts[d['task_id']]+=int(good(d))
            task_counts.setdefault(split,{})[model]=dict(counts)
        paired=read(R/('hard7-vs-broad51-'+split+'.json'))
        for a in ARMS:assert paired['models'][a]['authoritative_rescore_drift']==0
    assert total==180
    completion['actual_new_benchmark_runs']=total;completion['actual_training_rollouts']=408
    write(R/'completion.json',completion)
    aggregate={a:metrics[a]['iid_test']['strict_successes']+metrics[a]['ood_test']['strict_successes'] for a in metrics}
    delta=aggregate['broad51']-aggregate['hard7']
    phrase='高于' if delta>0 else '等于' if delta==0 else '低于'
    conclusion=f"同起点、各51步：完整51任务池的重复IID/OOD严格成功 {aggregate['broad51']}/66，{phrase}7任务池的 {aggregate['hard7']}/66；SFT参考为 {aggregate['sft']}/66。Dev分别为 SFT {metrics['sft']['dev']['strict_successes']}/24、7任务池 {metrics['hard7']['dev']['strict_successes']}/24、51任务池 {metrics['broad51']['dev']['strict_successes']}/24。"
    caveat='这是单种子、已观察测试集上的任务分布对照，不证明稳健或盲测泛化收益。'
    if aggregate['broad51']<aggregate['sft']:conclusion+=' 扩大任务池尚未超过SFT参考。'
    changed=[]
    for split in SPLITS:
        for t in sorted(task_counts[split]['sft']):
            if task_counts[split]['hard7'][t]!=task_counts[split]['broad51'][t]:
                item={'split':split,'task_id':t,'strict_counts':{a:task_counts[split][a][t] for a in metrics},'trials':3}
                for a in metrics:
                    traces=[json.loads(l) for l in (R/(a+'-'+split)/'traces.jsonl').read_text().splitlines()]
                    first=next(x for x in traces if x['task_id']==t and x['trial']==0)
                    item.setdefault('trial0_behavior',{})[a]=[{k:h.get(k) for k in ['thought','action']} for h in first['history']]
                changed.append(item)
    write(R/'task-delta-audit.json',{'strict_counts':task_counts,'between_arm_changed_tasks':changed,'interpretation':'Observed behavioral examples, not isolated proof of refusal-task-prior causality; no further training based on these traces.'})
    write(R/'mainline-outcome.json',{'plain_chinese':conclusion,'caveat':caveat,'metrics':metrics,'reused_test_aggregate_strict':aggregate,'broad_minus_hard_count':delta,'between_arm_changed_tasks':len(changed),'default_reference_model':plan['source_model'],'no_model_promoted':True,'one_rl_seed_only':True,'test_reuse_disclosed':True})
    table=[]
    for split in SPLITS:
        cells=[f"{metrics[a][split]['strict_successes']}/{metrics[a][split]['runs']} (pass³ {metrics[a][split]['pass3_tasks']}/{metrics[a][split]['tasks']})" for a in metrics]
        table.append('| '+split+' | '+' | '.join(cells)+' |')
    costs=[]
    for a,x in training.items():costs.append(f"- {a}: pool{x['pool_size']}, actual coverage{x['actual_task_coverage']},204 rollouts/51 groups, informative{x['informative_groups']}, clipped{x['clipped_rollouts']}, blocked{x['blocked_rollouts']}/204; process{x['process_seconds']:.2f}s, Torch allocation{x['torch_allocated_peak_gib']:.2f}GiB, sampled device max{x['sampled_device_max_mib']}MiB. Adapter{x['candidate_sha256']}, changed{x['changed_tensors']}tensors.")
    doc=f'''# Round006 — task-distribution comparison (2026-10-03 America/Los_Angeles)

{conclusion}

{caveat} SFT remains the reference; no model promotion. This directly tests one proposed data-distribution change after Round005's negative GRPO result.

## Pre-registered comparison

Both arms start from identical full SFT1, seed43, lr1e-5, G4/batch4, temperature1, six turns,2048 completion, GRPO/group/token IS/beta.04/truncation mask, no curriculum/canary/automatic verifier. Both run51 optimizer steps with existing allow_zero_variance option, disabling sampled saturation early stop for equal budgets. No learner, reward or prompt edits. Seven-task arm controls the full51-task arm. Equal steps and rollouts, not equal informative groups, policy tokens or compute. One dataset changes sampling/RNG consumption; same seed does not pair identical randomness.

Pools use only existing train parents, with no held-out overlap. Full pool reduces infeasible-task prior from3/7 to3/51; observed rollout exposure is recorded below. This compares the whole distribution, not an isolated refusal-task ratio or guaranteed capability preservation. Zero reward variance can mean no relative-advantage signal; positive mean reward is not validation.

Both training arms use the same empty scratch path. Evaluation keeps baseline prompts,bf16,seed45,3 trials,matching session prefix,identical empty Round005 benchmark scratch path. Round005 SFT's90 runs are reused with locked hashes;180 new model evaluations executed. All comparisons independently rescored against frozen oracles,zero strict drift and exceptions. All final candidates frozen before Round006 evaluation,no within-round test tuning.

## Results

| Split | SFT reused | hard7,51steps | broad51,51steps |
| --- | --- | --- | --- |
{chr(10).join(table)}

pass³ means all3 runs strictly correct,not pass@3. IID/OOD were observed in Round005 and are explicitly reused development/regression tests. These results do not establish blind confirmation; one training seed and small task sets remain limitations. Prior Round005 35-step result is context, not an equal-budget control for this experiment.

## Behavioral evidence

Between-arm strict-count changes occur in {len(changed)} tasks, archived in task-delta-audit.json. On this run, Dev search_kw_agentic_rl and IID search_kw_rag each change from hard7 0/3 to broad51 3/3; other task strict counts are unchanged. Hard7 trial0 prematurely FINISHes claiming a missing paper reference, although the request is keyword search. Broad51 calls search_arxiv_papers with query,max_results5,days30. This is a narrow recovery in keyword retrieval, not broad proof of improvement or an isolated causal test of infeasible-task ratio. SFT Dev is1/3 and IID3/3 on these tasks.

## Actual training and costs

{chr(10).join(costs)}

Only GPU4 used;5–7 reserved. GPU memory sampled every5seconds,not continuous physical peak. Training rollout summaries mix changing policies and are not frozen-model accuracy estimates.

## Mainline and next action

Completed reproducible data/environment -> full SFT -> real GRPO -> first IID/OOD -> controlled data-distribution ablation. Improvement still requires replicated gains and an independent evaluation design. Select the next experiment from these task-level outcomes: if broader data helps, replicate it and test on a new frozen set; if it only reduces regression or still fails, retain SFT and change one capability-preservation learning choice. Avoid another round centered on local logging repairs.

The world is curated real-paper extractive-caption tools. The author's missing VLM, original snapshot and weights are not recovered; do not claim exact original reproduction. No commits or pushes.
'''
    Path('docs/reproduction_round_006.md').write_text(doc);(R/'reproduction_round_006.md').write_text(doc)
    write(R/'final-source-manifest.json',{'source_sha256':{p:sha(p) for p in plan['sha256']},'finalizer_sha256':sha('scripts/finalize_round006_evidence.py'),'gpu_indices':[4],'actual_new_benchmark_runs':180,'reused_sft_runs':90,'actual_training_rollouts':408,'test_reuse_disclosed':True,'no_within_round_test_tuning':True,'original_exact_reproduction_complete':False})
    patch=subprocess.check_output(['git','diff','--binary','HEAD'])
    for p in subprocess.check_output(['git','ls-files','--others','--exclude-standard'],text=True).splitlines():
        if p.startswith(('scripts/','docs/','AgenticArxiv/tests/')) and Path(p).is_file():
            d=subprocess.run(['git','diff','--no-index','--binary','--','/dev/null',p],stdout=subprocess.PIPE)
            assert d.returncode in (0,1);patch+=d.stdout
    (R/'changes.patch').write_bytes(patch)
    subprocess.run(['git','diff','--check'],check=True)
    print(conclusion+' '+caveat)

if __name__=='__main__':main()
