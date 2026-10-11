#!/usr/bin/env python3
"""Close frozen real-error/replay experiment and all matched comparisons."""
from pathlib import Path
from datetime import datetime
import collections,json,subprocess
from run_round014_experiment import R,OLD,ARMS,CONTROLS,MODELS,PAIRS,GROUPS,read,sha,write,leaf
def good(d):return d['completed'] and d['tool_accurate'] and d['arg_score']==1 and d['ref_score']==1 and d['parse_fail']==0 and d['tool_fail']==0 and d['terminal_semantics_accurate']
def main():
    plan=read(R/'experiment-plan.json')
    for p,h in {**plan['sha256'],**plan['reused_evidence_sha256']}.items():assert sha(p)==h,p
    assert read(R/'certified-training-data-manifest.json')['support_gate_passed']
    start=min(datetime.fromisoformat(read(R/(a+'-train-job.json'))['started_at_utc']).timestamp() for a in ARMS)
    eval_start=min(datetime.fromisoformat(read(p)['started_at_utc']).timestamp() for p in R.glob('*-job.json') if 'train-job' not in p.name)
    assert (R/'fresh-tasks.json').stat().st_mtime<=(R/'experiment-plan.json').stat().st_mtime<=start
    from safetensors.torch import load_file
    import torch
    source=load_file(plan['evaluation_models']['sft']+'/adapter_model.safetensors');training={};chronology={}
    for a in ARMS:
        m=read(R/(a+'-training-manifest.json'));f=read(R/(a+'-candidate-freeze.json'));j=read(R/(a+'-train-job.json'))
        assert j['exit_code']==0 and j['gpu_index']==4 and m['actual_steps']==111 and m['dataset_rows']==443 and m['full_dataset_epochs']==1
        assert m['source_adapter_sha256']==plan['source_adapter_sha256'] and m['preference_data_sha256']==sha('data/repro_round014/certified_combined.jsonl')
        assert m['effective_batch']==4 and m['lr']==5e-6 and m['truncation_count']==0 and m['reference_equal_sft_and_frozen_tensors']==392
        assert m['dpo_loss_type']==(['sigmoid','sft'] if a.startswith('real') else ['sft'])
        assert m['dpo_loss_weights']==([1.,1.] if a.startswith('real') else [1.])
        assert sha(Path(f['model'])/'adapter_model.safetensors')==f['candidate_sha256']
        after=load_file(f['model']+'/adapter_model.safetensors');ref=load_file(f['model']+'/ref/adapter_model.safetensors')
        assert set(source)==set(after)==set(ref) and len(source)==392 and all(torch.equal(source[k],ref[k]) for k in source)
        assert sum(not torch.equal(source[k],after[k]) for k in source)==392
        finished=datetime.fromisoformat(j['finished_at_utc']).timestamp();frozen=datetime.fromisoformat(f['frozen_at_utc']).timestamp();assert finished<=frozen<=eval_start
        chronology[a]={'finished':finished,'frozen':frozen,'before_all_round014_inference':True}
        training[a]={**{k:m[k] for k in ['objective','seed','actual_steps','dataset_rows','effective_batch','lr','train_loss','trainer_elapsed_seconds','torch_peak_gib','reference_equal_sft_and_frozen_tensors','dpo_loss_type','dpo_loss_weights']},'changed_tensors':392,'process_seconds':j['elapsed_seconds'],'sampled_device_max_mib':j['sampled_max_device_used_mib'],'candidate_sha256':f['candidate_sha256']}
    write(R/'training-audit.json',training);write(R/'freeze-chronology-audit.json',{'first_training_start':start,'first_round014_eval_start':eval_start,'candidate_freezes':chronology})
    metrics={a:{} for a in MODELS};task_counts={};new=old=0
    pools={'dev':read('data/splits/v3_81.json')['split']['dev']}
    for prefix,p in [('previous',OLD),('fresh',R)]:pools.update({prefix+'-'+g:ids for g,ids in read(p/'fresh-splits.json')['split'].items()})
    for group,ids in pools.items():
        rescored={}
        for base,candidate in PAIRS:
            for model,m in read(R/(base+'-vs-'+candidate+'-'+group+'.json'))['models'].items():
                assert m['authoritative_rescore_drift']==0
                if model in rescored:assert rescored[model]==m
                rescored[model]=m
        assert set(rescored)==set(MODELS)
        for a in MODELS:
            d=R/leaf(a,group);summary=read(d/'summary.json');details=summary['details']
            assert summary['error_count']==0 and len(details)==len(ids)*3 and summary['sample_count']==len(details)
            assert {x['task_id'] for x in details}==set(ids) and len({(x['task_id'],x['trial']) for x in details})==len(details)
            if a in CONTROLS and (group=='dev' or group.startswith('previous')):old+=len(details)
            else:new+=len(details);assert read(R/(d.name+'-job.json'))['exit_code']==0
            m=rescored[a];success=sum(good(x) for x in details);assert success==round(m['strict_success_rate']*m['runs'])
            metrics[a][group]={**m,'strict_successes':success,'false_finish_count':sum(x['false_finish'] for x in details),'pass3_tasks':round(m['reliability']['pass^3']*m['tasks'])}
            counts=collections.Counter()
            for x in details:counts[x['task_id']]+=int(good(x))
            task_counts.setdefault(group,{})[a]=dict(counts)
    assert new==822 and old==270
    totals={p:{a:sum(metrics[a][p+'-'+g]['strict_successes'] for g in ['iid_test','ood_test']) for a in MODELS} for p in ['previous','fresh']}
    effects={p:{str(s):{b:totals[p]['real'+str(s)]-totals[p][b] for b in ['sft','ct'+str(s),'replay'+str(s)]} for s in [43,44]} for p in totals}
    fresh=totals['fresh'];extra=all(effects['fresh'][str(s)]['replay'+str(s)]>0 for s in [43,44]);earlier=all(effects['fresh'][str(s)]['ct'+str(s)]>0 and effects['fresh'][str(s)]['sft']>0 for s in [43,44]);dev_kept=all(metrics['real'+str(s)]['dev']['strict_successes']>=metrics['sft']['dev']['strict_successes'] for s in [43,44])
    conclusion=f"新22请求：原SFT {fresh['sft']}/66；旧继续SFT {fresh['ct43']}/66、{fresh['ct44']}/66；同数据继续SFT {fresh['replay43']}/66、{fresh['replay44']}/66；真实偏好＋回放 {fresh['real43']}/66、{fresh['real44']}/66。"
    conclusion+=('两个种子均超过同数据监督及旧控制且保留旧Dev成绩；仍需更强独立确认。' if extra and earlier and dev_kept else '未满足预先声明的完整收益标准，保留全部结果和原SFT参考，不调系数或选最佳种子。')
    write(R/'mainline-outcome.json',{'plain_chinese':conclusion,'metrics':metrics,'strict_totals':totals,'real_minus_controls':effects,'both_seeds_beat_matched_supervised':extra,'both_seeds_beat_earlier_controls':earlier,'old_dev_source_level_retained':dev_kept,'success_rule_met':extra and earlier and dev_kept,'new_runs':new,'reused_development_runs':old,'no_automatic_model_promotion':True})
    write(R/'task-level-audit.json',{'strict_counts':task_counts})
    changed=[]
    for group in pools:
        for task in sorted(task_counts[group]['sft']):
            counts={a:task_counts[group][a][task] for a in MODELS};ff={a:sum(x['false_finish'] for x in read(R/leaf(a,group)/'summary.json')['details'] if x['task_id']==task) for a in MODELS}
            if len(set(counts.values()))==len(set(ff.values()))==1:continue
            changed.append({'group':group,'task_id':task,'strict_counts':counts,'false_finish_counts':ff,'raw_trials':{a:[json.loads(s) for s in (R/leaf(a,group)/'traces.jsonl').read_text().splitlines() if json.loads(s)['task_id']==task] for a in MODELS}})
    write(R/'behavior-audit.json',{'changed_tasks':changed,'false_finish_scope':'Includes omitted steps, refusal and completion after failed tools; not exclusive false-refusal rate.'})
    write(R/'completion.json',{'state':'completed','actual_new_benchmark_runs':new,'actual_reused_development_runs':old,'actual_new_optimizer_steps':444,'actual_training_row_exposures':1772,'no_within_round_test_tuning':True})
    (R/'reproduction_round_014.md').write_text('# 第014轮：真实错误偏好与广泛回放对照\n\n'+conclusion+'\n\n45可审计模型错误＋398专家回放，两个种子，同443chosen数据/一次全遍历111步。模型可见状态重构为Benchmark提示，不宣称原生GRPO token逐字复原。仅GPU4，不使用5–7。已知caption世界与已观察留出父任务，作者完整VLM实验仍未复现。\n')
    subprocess.run(['git','diff','--check'],check=True)
    write(R/'final-source-manifest.json',{'frozen_sha256':{p:sha(p) for p in plan['sha256']},'git_diff_check':'passed','scoring_changed':False})
    print(conclusion,flush=True)
if __name__=='__main__':main()
