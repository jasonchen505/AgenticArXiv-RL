#!/usr/bin/env python3
"""Audit both seeds on the new test without treating trials as training seeds."""
from pathlib import Path
import collections
import hashlib
import json
import subprocess
from datetime import datetime

R=Path('artifacts/reproduction/round-007-2026-10-04')
MODELS=['sft','hard43','broad43','hard44','broad44']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def good(d):return bool(d['completed'] and d['tool_accurate'] and d['arg_score']==1 and d['ref_score']==1 and d['parse_fail']==0 and d['tool_fail']==0 and d['terminal_semantics_accurate'])

def main():
    completion=read(R/'completion.json');assert completion['state']=='completed'
    plan=read(R/'experiment-plan.json');fresh=read(R/'fresh-splits.json')['split']
    for p,h in plan['sha256'].items():assert sha(p)==h,p
    train_start=min(datetime.fromisoformat(read(R/(a+'-train-job.json'))['started_at_utc']).timestamp() for a in ['hard44','broad44'])
    first_eval=min(datetime.fromisoformat(read(p)['started_at_utc']).timestamp() for p in R.glob('*-job.json') if 'train-job' not in p.name)
    first_fresh=min(datetime.fromisoformat(read(p)['started_at_utc']).timestamp() for p in R.glob('fresh-*-job.json'))
    assert (R/'fresh-tasks.json').stat().st_mtime <= (R/'experiment-plan.json').stat().st_mtime <= train_start
    chronology={'fresh_tasks_mtime_unix':(R/'fresh-tasks.json').stat().st_mtime,'plan_mtime_unix':(R/'experiment-plan.json').stat().st_mtime,'first_train_start_unix':train_start,'first_round007_eval_start_unix':first_eval,'first_fresh_eval_start_unix':first_fresh,'candidates':{},'metadata_note':'Raw candidate freeze uses a copied Round006 boolean field name; actual Round007 chronology is verified here. Raw records retained.'}
    for a in ['hard44','broad44']:
        frozen_at=(R/(a+'-candidate-freeze.json')).stat().st_mtime
        finished=datetime.fromisoformat(read(R/(a+'-train-job.json'))['finished_at_utc']).timestamp()
        assert finished <= frozen_at <= first_eval
        chronology['candidates'][a]={'train_finished_unix':finished,'freeze_mtime_unix':frozen_at,'before_round007_evaluation':True}
    write(R/'freeze-chronology-audit.json',chronology)
    training={}
    for a in ['hard44','broad44']:
        m=read(R/(a+'-training-manifest.json'));f=read(R/(a+'-candidate-freeze.json'));job=read(R/(a+'-train-job.json'))
        assert m['actual_steps']==51 and not m['reward_variance_guard']['enabled'] and job['exit_code']==0
        assert sha(Path(f['model'])/'adapter_model.safetensors')==f['candidate_sha256']
        samples=[json.loads(l) for l in (R/(a+'-rollout-traces.jsonl')).read_text().splitlines()]
        groups=collections.defaultdict(list)
        for s in samples:groups[(s['batch_index'],s['group_index'])].append(s)
        assert len(samples)==204 and len(groups)==51 and all(len(v)==4 for v in groups.values())
        counts=collections.Counter(s['task_id'] for s in samples);assert set(counts)<=set(plan['training_tasks'][a])
        training[a]={'steps':51,'samples':204,'groups':51,'task_coverage':len(counts),'pool_size':len(plan['training_tasks'][a]),'informative_groups':sum(v[0]['group_reward_std']>0 for v in groups.values()),'blocked_rollouts':sum(n for t,n in counts.items() if t.startswith('infeasible_')),'clipped':sum(s['anomalies']['clipped'] for s in samples),'process_seconds':job['elapsed_seconds'],'torch_allocated_peak_gib':m['peak_vram_gib'],'sampled_device_max_mib':job['sampled_max_device_used_mib'],'changed_tensors':f['changed_tensors'],'candidate_sha256':f['candidate_sha256'],'task_rollout_counts':dict(counts)}
    write(R/'training-audit.json',training)
    for a,h in plan['reference_model_hashes'].items():assert sha(Path(plan['evaluation_models'][a])/'adapter_model.safetensors')==h
    metrics={a:{} for a in MODELS};task_counts={};new_count=0;reused_count=0
    pools={'dev':read('data/splits/v3_81.json')['split']['dev'],**{'fresh-'+k:v for k,v in fresh.items()}}
    for group,ids in pools.items():
        for a in MODELS:
            directory=R/(a+'-dev') if group=='dev' else R/('fresh-'+a+'-'+group.removeprefix('fresh-'))
            summary=read(directory/'summary.json')
            assert summary['error_count']==0 and summary['sample_count']==len(ids)*3
            assert {d['task_id'] for d in summary['details']}==set(ids)
            assert len({(d['task_id'],d['trial']) for d in summary['details']})==len(ids)*3
            assert all(d['trial'] in [0,1,2] for d in summary['details'])
            if group=='dev' and a in ['sft','hard43','broad43']:
                for f in ['summary.json','traces.jsonl','execution_protocol.json']:assert sha(directory/f)==plan['dev_evidence_hashes'][f'{a}/{f}']
                reused_count+=summary['sample_count']
            else:
                assert read(R/(directory.name+'-job.json'))['exit_code']==0
                new_count+=summary['sample_count']
            comp=read(R/('sft-vs-hard43-'+group+'.json')) if a=='sft' else read(R/('sft-vs-'+a+'-'+group+'.json'))
            x=comp['models'][a];assert x['authoritative_rescore_drift']==0
            count=sum(good(d) for d in summary['details'])
            assert count==round(x['strict_success_rate']*x['runs'])
            metrics[a][group]={**x,'strict_successes':count,'pass3_tasks':round(x['reliability']['pass^3']*x['tasks'])}
            per=collections.Counter()
            for d in summary['details']:per[d['task_id']]+=int(good(d))
            task_counts.setdefault(group,{})[a]=dict(per)
        for seed in [43,44]:
            pair=read(R/(f'hard{seed}-vs-broad{seed}-'+group+'.json'))
            assert all(m['authoritative_rescore_drift']==0 for m in pair['models'].values())
    assert new_count==378 and reused_count==72
    totals={a:sum(metrics[a]['fresh-'+s]['strict_successes'] for s in ['iid_test','ood_test']) for a in MODELS}
    effects={str(seed):totals['broad'+str(seed)]-totals['hard'+str(seed)] for seed in [43,44]}
    stable=all(d>0 for d in effects.values())
    conclusion=f"新冻结22任务：SFT {totals['sft']}/66；seed43难例池 {totals['hard43']}/66、完整池 {totals['broad43']}/66；seed44难例池 {totals['hard44']}/66、完整池 {totals['broad44']}/66。"
    conclusion+=('完整任务池在两个种子均高于难例池。' if stable else '完整任务池相对难例池的优势没有在两个种子均保持。')
    above_sft=all(totals['broad'+str(s)]>totals['sft'] for s in [43,44])
    conclusion+=('两个完整池候选均高于SFT，但样本与种子仍少。' if above_sft else '尚不能建立两个种子均超过SFT的收益。')
    caveat='新参数契约在模型输出前冻结，但基于已观察留出父任务、同一已知论文世界。原OOD分组有1/4完整工具顺序已在训练中出现；两个RL种子和22任务不足以证明广泛模板泛化。'
    completion.update(actual_new_benchmark_runs=new_count,actual_reused_dev_runs=reused_count,actual_training_rollouts=408)
    write(R/'completion.json',completion)
    write(R/'mainline-outcome.json',{'plain_chinese':conclusion,'caveat':caveat,'metrics':metrics,'fresh_strict_totals':totals,'broad_minus_hard_counts_by_seed':effects,'both_seeds_broad_better_than_hard':stable,'both_broad_seeds_above_sft':above_sft,'default_reference_model':plan['source_model'],'no_automatic_model_promotion':True,'rl_seeds':[43,44],'one_sft_seed':True,'fresh_new_parameter_contracts':22,'no_fresh_output_based_tuning':True})
    write(R/'task-level-audit.json',{'strict_counts':task_counts,'meaning':'Per-task3trial counts; do not treat repeated trials as independent training seeds.'})
    changed=[]
    for group in ['fresh-iid_test','fresh-ood_test']:
        for t in sorted(task_counts[group]['sft']):
            counts={a:task_counts[group][a][t] for a in MODELS}
            if len(set(counts.values()))==1:continue
            item={'task_id':t,'group':group,'strict_counts':counts,'raw_trials':{}}
            for a in MODELS:
                directory=R/('fresh-'+a+'-'+group.removeprefix('fresh-'))
                traces=[json.loads(l) for l in (directory/'traces.jsonl').read_text().splitlines()]
                item['raw_trials'][a]=[x for x in traces if x['task_id']==t]
            changed.append(item)
    write(R/'behavior-audit.json',{'tasks_with_different_strict_counts':changed,'count':len(changed),'other_tasks_with_identical_strict_counts':22-len(changed),'limit':'Identical strict counts do not imply identical outputs,partial scores or costs. No criterion changed after reading traces.'})
    table=[]
    for group in pools:
        cells=[f"{metrics[a][group]['strict_successes']}/{metrics[a][group]['runs']} (pass³{metrics[a][group]['pass3_tasks']}/{metrics[a][group]['tasks']})" for a in MODELS]
        table.append('| '+group+' | '+' | '.join(cells)+' |')
    doc=f'''# Round007 — seed replication and new frozen tests

2026-10-04 America/Los_Angeles

{conclusion}

{caveat}

## Protocol

Seed44 independently starts from the identical SFT1,hard7/broad51 train pools,51steps/204rollouts each. Same lr1e-5,G4/batch4,temperature1,6turns/2048completion,GRPO/group/tokenIS/beta.04/truncationmask,no curriculum,no sampled variance early stop,no prompt/reward/learner edits. Seed43 uses unchanged Round006 final models. One SFT seed42,two RL seeds43/44. Hashes frozen before new model output,no Dev checkpoint selection or test-based tuning.

New22 tasks are deterministic parameter recombinations of existing held-out parents:18 IID and4 OOD. They have no train-parent overlap and no exact tool/setup/terminal contract overlap with original81 or SFT137 semantic instances. Fresh gold22/22,same curated caption snapshot. Expected arguments and oracles remain outside agent inputs; existing BenchmarkRunner uses request and visible setup as before. No new evaluation dataset enters training.

All5 models execute fresh tests,330runs;seed44 Dev adds48runs. Existing Dev for SFT/seed43 reuses72hash-locked runs. Total378new/72reused,zero exceptions or strict authoritative rescore drift. Paired same baseline prompt,bf16,seed45,prefix and empty scratch. New task IDs define new shared per-task random streams. All final candidates frozen before evaluation. Semantic gold is not model success.

## Metrics

| Split | SFT | hard43 | broad43 | hard44 | broad44 |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(table)}

Fresh and previous tasks have different parameters/instructions,so do not compare their absolute rates as if task difficulty were identical. pass³ means3/3 strict success,not pass@3. Seeds share the sameSFT source and test set; counts are correlated and not hundreds of independent tasks. Structural program audit finds16/22fresh tasks share a full tool sequence with training,including1/4legacyOOD. Exact request/setup/argument contracts still differ. No tasks were removed based on this audit or model outcomes. Treat original OOD labels as legacy groups,not proof all programs are unseen. See template-scope-audit.json. Fresh tests become observed after this run; further tuning must disclose reuse.

## Narrow behavioral difference

Only{len(changed)}new task has different strict counts across models;the other{22-len(changed)}have identical strict-count outcomes,not necessarily identical outputs. The changed case requests paper9 from a4-paper session. SFT gives the correct invalid-index explanation2/3;hard44 does3/3. hard43/broad43 refuse with the wrong reason,claiming a missing arXivID0/3. broad44 calls download with ref9 and then claims completion0/3. The scoring criterion was unchanged and this is a real semantic difference,not a new regex reinterpretation. See behavior-audit.json. There is no broad tool-capability improvement in these results; next work should isolate termination-semantics learning while retaining the SFT reference and equal budgets.

## Resources and mainline

{json.dumps(training,ensure_ascii=False,indent=2)}

OnlyGPU4 used,GPUs5–7reserved. Torch allocation differs from5second sampled device usage; sampled max is not continuous physical peak. Installed TRL clones the initial SFT adapter as ref for this PEFT path. Saved references of all4GRPO models are CPU-verified392/392tensor-identical toSFT; this rules out a missing SFT reference,not insufficient KL regularization. See reference-policy-audit.json. Core-source hashes unchanged. No broad unit-test campaign; validation is real training and complete paired task execution,weight/input hashes,CPU gold,authoritative rescoring,compile/diff and HTML browser checks.

Mainline now covers reproduction environment/data,full597stepSFT,realGRPO,negative held-out result,controlled task-distribution ablation,and second-seed/new-input confirmation. RetainSFT reference until replicated useful gains support promotion. Choose one capability-preservation learner change if data-only recovery remains belowSFT; if confirmation fails,report instability before adding complexity. Original author's VLM/snapshot/weights remain missing; this is not exact author reproduction. No commits or pushes.
'''
    Path('docs/reproduction_round_007.md').write_text(doc);(R/'reproduction_round_007.md').write_text(doc)
    write(R/'final-source-manifest.json',{'source_sha256':{p:sha(p) for p in plan['sha256']},'finalizer_sha256':sha('scripts/finalize_round007_evidence.py'),'allowed_gpus':[4],'actual_new_benchmark_runs':378,'reused_dev_runs':72,'actual_training_rollouts':408,'no_test_based_tuning':True,'original_author_exact_reproduction_complete':False})
    patch=subprocess.check_output(['git','diff','--binary','HEAD'])
    for p in subprocess.check_output(['git','ls-files','--others','--exclude-standard'],text=True).splitlines():
        if p.startswith(('scripts/','docs/','AgenticArxiv/tests/')) and Path(p).is_file():
            d=subprocess.run(['git','diff','--no-index','--binary','--','/dev/null',p],stdout=subprocess.PIPE)
            assert d.returncode in [0,1];patch+=d.stdout
    (R/'changes.patch').write_bytes(patch);subprocess.run(['git','diff','--check'],check=True)
    print(conclusion+' '+caveat)

if __name__=='__main__':main()
