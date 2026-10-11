#!/usr/bin/env python3
"""Verify realized reward gates, paired evaluations and all frozen inputs."""
from pathlib import Path
from datetime import datetime
import collections,hashlib,json,subprocess,sys
R=Path('artifacts/reproduction/round-008-2026-10-04')
MODELS=['sft','hard43','gate43','hard44','gate44'];ARMS=['gate43','gate44']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def good(d):return bool(d['completed'] and d['tool_accurate'] and d['arg_score']==1 and d['ref_score']==1 and d['parse_fail']==0 and d['tool_fail']==0 and d['terminal_semantics_accurate'])
def leaf(a,g):return a+'-dev' if g=='dev' else g.split('-')[0]+'-'+a+'-'+g.split('-',1)[1]

def main():
    assert read(R/'completion.json')['state']=='completed'
    plan=read(R/'experiment-plan.json')
    for p,h in {**plan['sha256'],**plan['reused_evidence_sha256']}.items():assert sha(p)==h,p
    for a,h in plan['reference_model_hashes'].items():assert sha(Path(plan['evaluation_models'][a])/'adapter_model.safetensors')==h
    sys.path.insert(0,str(Path('AgenticArxiv').resolve()))
    from benchmark.metrics import extract_metrics,is_strict_success
    from rl.reward import RewardCalculator
    oracles={x['id']:x for x in read('artifacts/reproduction/round-002-2026-10-02/task-oracles.json')}
    calc=RewardCalculator(curriculum_steps=0);training={}
    first_eval=min(datetime.fromisoformat(read(p)['started_at_utc']).timestamp() for p in R.glob('*-job.json') if 'train-job' not in p.name)
    first_train=min(datetime.fromisoformat(read(R/(a+'-train-job.json'))['started_at_utc']).timestamp() for a in ARMS)
    assert (R/'fresh-tasks.json').stat().st_mtime <= (R/'experiment-plan.json').stat().st_mtime <= first_train
    chronology={'fresh_tasks_mtime_unix':(R/'fresh-tasks.json').stat().st_mtime,'plan_mtime_unix':(R/'experiment-plan.json').stat().st_mtime,'first_train_start_unix':first_train,'first_eval_start_unix':first_eval,'candidate_freezes':{}}
    for a in ARMS:
        job=read(R/(a+'-train-job.json'));manifest=read(R/(a+'-training-manifest.json'));frozen=read(R/(a+'-candidate-freeze.json'))
        assert job['exit_code']==0 and manifest['actual_steps']==51 and manifest['seed']==int(a[-2:]) and manifest['reward_objective']=='blocked_reason_v1'
        assert sha(Path(frozen['model'])/'adapter_model.safetensors')==frozen['candidate_sha256']
        finished=datetime.fromisoformat(job['finished_at_utc']).timestamp();freezetime=(R/(a+'-candidate-freeze.json')).stat().st_mtime
        assert finished<=freezetime<=first_eval
        chronology['candidate_freezes'][a]={'train_finished_unix':finished,'freeze_mtime_unix':freezetime,'before_all_evaluation':True}
        samples=[json.loads(l) for l in (R/(a+'-rollout-traces.jsonl')).read_text().splitlines()];assert len(samples)==204
        groups=collections.defaultdict(list);counts=collections.Counter(x['task_id'] for x in samples);gate_counts=collections.Counter()
        for x in samples:
            groups[(x['training_step'],x['group_index'])].append(x)
            audit=x['trajectory']['training_reward_audit']
            legacy,metric=calc.compute_reward_breakdown(oracles[x['task_id']],x['trajectory'])
            triggered=metric.termination_type=='FINISH' and metric.terminal_semantics_applicable and not metric.terminal_semantics_accurate
            expected=min(legacy.total,0.) if triggered else legacy.total
            assert legacy.total==audit['legacy_reward'] and expected==x['reward']==audit['effective_reward']==x['reward_breakdown']['total']
            assert triggered==audit['gate_triggered'] and (expected!=legacy.total)==audit['reward_changed']
            gate_counts['triggered']+=triggered;gate_counts['reward_changed']+=audit['reward_changed']
            gate_counts['strict_failures']+=not is_strict_success(metric)
        assert len(groups)==51 and all(len(v)==4 for v in groups.values()) and set(counts)==set(plan['training_tasks'][a])
        previous=Path('artifacts/reproduction')/('round-006-2026-10-03/hard7-rollout-traces.jsonl' if a=='gate43' else 'round-007-2026-10-04/hard44-rollout-traces.jsonl')
        prior=[json.loads(l) for l in previous.read_text().splitlines()]
        initial_equal=sum(x['task_id']==y['task_id'] and x['raw_assistant_turns']==y['raw_assistant_turns'] for x,y in zip(samples[:4],prior[:4]))
        training[a]={'steps':51,'samples':204,'groups':51,'task_coverage':len(counts),'pool_size':7,'informative_groups':sum(v[0]['group_reward_std']>0 for v in groups.values()),'gate_counts':dict(gate_counts),'clipped':sum(x['anomalies']['clipped'] for x in samples),'process_seconds':job['elapsed_seconds'],'torch_peak_gib':manifest['peak_vram_gib'],'sampled_device_max_mib':job['sampled_max_device_used_mib'],'changed_tensors':frozen['changed_tensors'],'ref_equal_sft_tensors':frozen['ref_equal_sft_tensors'],'first_preupdate_control_rollouts_identical':initial_equal,'task_rollout_counts':dict(counts),'candidate_sha256':frozen['candidate_sha256']}
    write(R/'training-audit.json',training);write(R/'freeze-chronology-audit.json',chronology)
    metrics={a:{} for a in MODELS};per_task={};new_count=old_count=0
    groups={'dev':read('data/splits/v3_81.json')['split']['dev']}
    for prefix,path in [('previous',Path('artifacts/reproduction/round-007-2026-10-04')),('fresh',R)]:groups.update({prefix+'-'+s:ids for s,ids in read(path/'fresh-splits.json')['split'].items()})
    for group,ids in groups.items():
        for a in MODELS:
            directory=R/leaf(a,group);summary=read(directory/'summary.json');assert summary['error_count']==0 and summary['sample_count']==len(ids)*3
            details=summary['details'];assert {d['task_id'] for d in details}==set(ids) and len({(d['task_id'],d['trial']) for d in details})==len(ids)*3
            if group=='dev' or group.startswith('previous'):
                if a in ['sft','hard43','hard44']:old_count+=len(details)
                else:new_count+=len(details);assert read(R/(directory.name+'-job.json'))['exit_code']==0
            else:new_count+=len(details);assert read(R/(directory.name+'-job.json'))['exit_code']==0
            comp=read(R/(('sft-vs-hard43-' if a=='sft' else 'sft-vs-'+a+'-')+group+'.json'))
            m=comp['models'][a];assert m['authoritative_rescore_drift']==0
            count=sum(good(d) for d in details);assert count==round(m['strict_success_rate']*m['runs'])
            metrics[a][group]={**m,'strict_successes':count,'pass3_tasks':round(m['reliability']['pass^3']*m['tasks'])}
            per=collections.Counter()
            for d in details:per[d['task_id']]+=int(good(d))
            per_task.setdefault(group,{})[a]=dict(per)
        for seed in [43,44]:assert all(x['authoritative_rescore_drift']==0 for x in read(R/(f'hard{seed}-vs-gate{seed}-'+group+'.json'))['models'].values())
    assert new_count==510 and old_count==270
    totals={prefix:{a:sum(metrics[a][prefix+'-'+s]['strict_successes'] for s in ['iid_test','ood_test']) for a in MODELS} for prefix in ['previous','fresh']}
    effects={prefix:{str(seed):totals[prefix]['gate'+str(seed)]-totals[prefix]['hard'+str(seed)] for seed in [43,44]} for prefix in totals}
    fresh=totals['fresh'];stable=all(effects['fresh'][str(s)]>0 for s in [43,44]);above=all(fresh['gate'+str(s)]>fresh['sft'] for s in [43,44])
    conclusion=f"新22任务：SFT {fresh['sft']}/66；旧奖励seed43 {fresh['hard43']}/66、seed44 {fresh['hard44']}/66；理由封顶seed43 {fresh['gate43']}/66、seed44 {fresh['gate44']}/66。"
    conclusion+=('理由封顶在两个种子均高于各自旧奖励对照。' if stable else '理由封顶没有在两个种子均稳定提高新任务严格成功。')
    conclusion+=('两个候选均高于SFT，仍需更大独立确认。' if above else '没有建立两个种子均超过SFT的收益，保留SFT参考。')
    write(R/'mainline-outcome.json',{'plain_chinese':conclusion,'metrics':metrics,'strict_totals':totals,'gate_minus_legacy':effects,'both_fresh_seeds_gain_vs_control':stable,'both_fresh_seeds_above_sft':above,'reference_model':plan['evaluation_models']['sft'],'no_automatic_model_promotion':True,'one_sft_seed':True,'rl_seeds':[43,44],'new_benchmark_runs':new_count,'reused_development_runs':old_count})
    write(R/'task-level-audit.json',{'strict_counts':per_task})
    changed=[]
    for group in groups:
        for task in sorted(per_task[group]['sft']):
            counts={a:per_task[group][a][task] for a in MODELS}
            if len(set(counts.values()))==1:continue
            traces={a:[json.loads(l) for l in (R/leaf(a,group)/'traces.jsonl').read_text().splitlines() if json.loads(l)['task_id']==task] for a in MODELS}
            changed.append({'group':group,'task_id':task,'strict_counts':counts,'raw_trials':traces})
    write(R/'behavior-audit.json',{'changed_tasks':changed,'scoring_unchanged':True})
    from benchmark.tasks_expanded import get_expanded_tasks
    sys.path.insert(0,str(Path('scripts').resolve()))
    from generate_parametric_sft_data import build_parametric_tasks
    train=set(read('data/splits/v3_81.json')['split']['train']);specs=[t for t in get_expanded_tasks() if t['id'] in train]+[d.spec.to_task() for d in build_parametric_tasks(include_t5=True)]
    programs={tuple(t['expected_tools']) for t in specs};newtasks=read(R/'fresh-tasks.json');ood=set(read(R/'fresh-splits.json')['split']['ood_test'])
    scope={'training_semantic_instances':len(specs),'fresh_tasks':22,'full_tool_program_in_train':sum(tuple(t['expected_tools']) in programs for t in newtasks),'legacy_ood_group_tasks':4,'legacy_ood_full_tool_program_in_train':sum(t['id'] in ood and tuple(t['expected_tools']) in programs for t in newtasks),'meaning':'New parameter contracts, known paper world and observed parents. Original OOD grouping does not guarantee unseen exact tool programs.'}
    write(R/'template-scope-audit.json',scope)
    completion=read(R/'completion.json');completion.update(actual_new_benchmark_runs=new_count,actual_reused_development_runs=old_count,actual_training_rollouts=408);write(R/'completion.json',completion)
    doc=f"# 第008轮：终止理由奖励对照\n\n{conclusion}\n\n仅改变训练奖励：blocked FINISH理由错误时封顶0，其余沿用旧奖励。原评测计分、模型提示不变；同一SFT、难例7任务、每组51步204rollout，RL seed43/44。旧奖励模型和270次开发记录复用，510次新评测，全部无异常、权威严格重计分漂移0。\n\n新测试在输出前冻结，标准22/22；同一已知论文世界和已观察父任务。{scope['full_tool_program_in_train']}/22工具程序已在训练出现，原OOD组{scope['legacy_ood_full_tool_program_in_train']}/4程序已出现。第007轮测试复用明确作为开发集。\n\n两个训练模型保存的ref均与SFT392/392张量相同。没有按评测选checkpoint、最好种子或删除案例。仅GPU4；GPU5–7不使用。原作者缺失VLM/快照/权重仍未恢复。\n"
    Path('docs/reproduction_round_008.md').write_text(doc);(R/'reproduction_round_008.md').write_text(doc)
    subprocess.run(['git','diff','--check'],check=True)
    with (R/'changes.patch').open('w') as file:subprocess.run(['git','diff'],stdout=file,check=True)
    write(R/'final-source-manifest.json',{'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'sha256':{p:sha(p) for p in plan['sha256']},'git_diff_check':'passed','evaluation_scoring_changed':False})
    print(json.dumps({'new_runs':new_count,'reused_runs':old_count,'fresh_totals':fresh,'effects':effects,'training':training},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
