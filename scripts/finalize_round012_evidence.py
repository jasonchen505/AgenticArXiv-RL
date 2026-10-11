#!/usr/bin/env python3
"""Independently close all model comparisons, immutable inputs and chronology."""
from pathlib import Path
from datetime import datetime
import collections,hashlib,json,subprocess
from run_round012_experiment import R,OLD,ARMS,CONTROLS,MODELS,PAIRS,GROUPS,read,sha,write,leaf
def good(d):return d['completed'] and d['tool_accurate'] and d['arg_score']==1 and d['ref_score']==1 and d['parse_fail']==0 and d['tool_fail']==0 and d['terminal_semantics_accurate']
def main():
    plan=read(R/'experiment-plan.json')
    for p,h in {**plan['sha256'],**plan['reused_evidence_sha256']}.items():assert sha(p)==h,p
    train_start=min(datetime.fromisoformat(read(R/(a+'-train-job.json'))['started_at_utc']).timestamp() for a in ARMS)
    eval_start=min(datetime.fromisoformat(read(p)['started_at_utc']).timestamp() for p in R.glob('*-job.json') if 'train-job' not in p.name)
    assert (R/'fresh-tasks.json').stat().st_mtime<=(R/'experiment-plan.json').stat().st_mtime<=train_start
    training={};chron={'first_train_start_unix':train_start,'first_eval_start_unix':eval_start,'fresh_tasks_mtime_unix':(R/'fresh-tasks.json').stat().st_mtime,'candidate_freezes':{}}
    from safetensors.torch import load_file
    import torch
    source=load_file(plan['evaluation_models']['sft']+'/adapter_model.safetensors')
    for a in ARMS:
        m=read(R/(a+'-training-manifest.json'));f=read(R/(a+'-candidate-freeze.json'));j=read(R/(a+'-train-job.json'))
        assert j['exit_code']==0 and m['actual_steps']==51 and m['seed']==int(a[-2:])
        assert m['source_adapter_sha256']==plan['source_adapter_sha256'] and m['preference_data_sha256']==sha('data/repro_round010/preferences.jsonl')
        assert m['dpo_loss_type']==['sigmoid','sft'] and m['dpo_loss_weights']==[1.,1.] and m['beta']==.1
        assert m['effective_batch']==4 and m['batch_size']==1 and m['gradient_accumulation_steps']==4 and m['lr']==5e-6
        assert m['dataset_rows']==448 and m['truncation_count']==0 and m['reference_equal_sft_and_frozen_tensors']==392
        assert sha(Path(f['model'])/'adapter_model.safetensors')==f['candidate_sha256']
        candidate=load_file(f['model']+'/adapter_model.safetensors');ref=load_file(f['model']+'/ref/adapter_model.safetensors')
        assert set(source)==set(candidate)==set(ref) and len(source)==392
        assert all(torch.equal(source[k],ref[k]) for k in source)
        assert sum(not torch.equal(source[k],candidate[k]) for k in source)==392
        finish=datetime.fromisoformat(j['finished_at_utc']).timestamp();frozen=datetime.fromisoformat(f['frozen_at_utc']).timestamp()
        assert finish<=frozen<=eval_start
        chron['candidate_freezes'][a]={'finished_unix':finish,'frozen_unix':frozen,'before_all_round012_inference':True}
        training[a]={**{k:m[k] for k in ['objective','seed','actual_steps','dataset_rows','effective_batch','lr','train_loss','trainer_elapsed_seconds','torch_peak_gib','reference_equal_sft_and_frozen_tensors','dpo_loss_type','dpo_loss_weights']},'changed_tensors':392,'process_seconds':j['elapsed_seconds'],'sampled_device_max_mib':j['sampled_max_device_used_mib'],'candidate_sha256':f['candidate_sha256']}
    write(R/'training-audit.json',training);write(R/'freeze-chronology-audit.json',chron)
    metrics={a:{} for a in MODELS};task_counts={};new=old=0
    pools={'dev':read('data/splits/v3_81.json')['split']['dev']}
    for prefix,path in [('previous',OLD),('fresh',R)]:pools.update({prefix+'-'+g:ids for g,ids in read(path/'fresh-splits.json')['split'].items()})
    for group,ids in pools.items():
        rescored={}
        for b,a in PAIRS:
            comp=read(R/(b+'-vs-'+a+'-'+group+'.json'))['models']
            for model,m in comp.items():
                assert m['authoritative_rescore_drift']==0
                if model in rescored:assert rescored[model]==m,(group,model)
                rescored[model]=m
        assert set(rescored)==set(MODELS)
        for a in MODELS:
            directory=R/leaf(a,group);s=read(directory/'summary.json');ds=s['details']
            assert s['error_count']==0 and s['sample_count']==len(ids)*3
            assert {d['task_id'] for d in ds}==set(ids) and len({(d['task_id'],d['trial']) for d in ds})==len(ids)*3
            if a in CONTROLS and (group=='dev' or group.startswith('previous')):old+=len(ds)
            else:new+=len(ds);assert read(R/(directory.name+'-job.json'))['exit_code']==0
            success=sum(good(d) for d in ds);m=rescored[a];assert success==round(m['strict_success_rate']*m['runs'])
            metrics[a][group]={**m,'strict_successes':success,'pass3_tasks':round(m['reliability']['pass^3']*m['tasks']),'false_finish_count':sum(d['false_finish'] for d in ds)}
            counts=collections.Counter()
            for d in ds:counts[d['task_id']]+=int(good(d))
            task_counts.setdefault(group,{})[a]=dict(counts)
    assert new==642 and old==450
    totals={p:{a:sum(metrics[a][p+'-'+g]['strict_successes'] for g in ['iid_test','ood_test']) for a in MODELS} for p in ['previous','fresh']}
    effects={p:{str(seed):{b:totals[p]['anchor'+str(seed)]-totals[p][b] for b in ['sft','ct'+str(seed),'dpo'+str(seed)]} for seed in [43,44]} for p in totals}
    fresh=totals['fresh'];beats_dpo=all(effects['fresh'][str(s)]['dpo'+str(s)]>0 for s in [43,44]);beats_ct=all(effects['fresh'][str(s)]['ct'+str(s)]>0 for s in [43,44]);beats_sft=all(effects['fresh'][str(s)]['sft']>0 for s in [43,44])
    conclusion=f"新22请求：原SFT {fresh['sft']}/66；继续SFT {fresh['ct43']}/66、{fresh['ct44']}/66；纯DPO {fresh['dpo43']}/66、{fresh['dpo44']}/66；DPO＋监督 {fresh['anchor43']}/66、{fresh['anchor44']}/66。"
    conclusion+=('两个种子均超过继续SFT与原SFT，仍需更强独立确认。' if beats_ct and beats_sft else '没有建立两个种子均超过继续SFT及原SFT的收益，保留原SFT参考。')
    conclusion+=('相比纯DPO的恢复不能单独当作超过监督对照的收益。' if beats_dpo and not beats_ct else '')
    write(R/'mainline-outcome.json',{'plain_chinese':conclusion,'metrics':metrics,'strict_totals':totals,'anchor_minus_controls':effects,'both_seeds_beat_pure_dpo':beats_dpo,'both_seeds_beat_continued_sft':beats_ct,'both_seeds_beat_source_sft':beats_sft,'reference_model':plan['evaluation_models']['sft'],'no_automatic_model_promotion':True,'new_runs':new,'reused_development_runs':old,'stop_synthetic_negative_objective_branch':not beats_ct})
    write(R/'task-level-audit.json',{'strict_counts':task_counts})
    changed=[]
    for group in pools:
        for t in sorted(task_counts[group]['sft']):
            counts={a:task_counts[group][a][t] for a in MODELS};ff={a:sum(d['false_finish'] for d in read(R/leaf(a,group)/'summary.json')['details'] if d['task_id']==t) for a in MODELS}
            if len(set(counts.values()))==1 and len(set(ff.values()))==1:continue
            raw={a:[json.loads(s) for s in (R/leaf(a,group)/'traces.jsonl').read_text().splitlines() if json.loads(s)['task_id']==t] for a in MODELS}
            changed.append({'group':group,'task_id':t,'strict_counts':counts,'premature_finish_counts':ff,'raw_trials':raw})
    write(R/'behavior-audit.json',{'changed_tasks':changed,'false_finish_definition':'Repository terminal metric includes missing required tools and may describe refusal, step omission or completion claims after failed tools; not exclusive incorrect-refusal rate.'})
    write(R/'completion.json',{'state':'completed','actual_new_benchmark_runs':new,'actual_reused_development_runs':old,'actual_new_optimizer_steps':102,'actual_new_training_prompt_exposures':408,'no_within_round_test_tuning':True})
    doc=f'# 第012轮：DPO＋监督能力保持对照\n\n{conclusion}\n\n只增加chosen token-mean监督损失，固定[1,1]与beta0.1；同448训练对、源SFT、51步、有效批量4、lr5e-6，两个种子。五个已冻结模型作对照。642次新评测、450次复用开发记录，零异常/权威重计分漂移。已知caption论文世界与观察过的留出父任务，未恢复作者VLM/快照/权重；不是新世界或完全未见程序泛化。只用GPU4，不用5–7，不选最佳种子或测试驱动checkpoint。\n'
    (R/'reproduction_round_012.md').write_text(doc);Path('docs/reproduction_round_012.md').write_text(doc)
    subprocess.run(['git','diff','--check'],check=True)
    write(R/'final-source-manifest.json',{'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'frozen_source_sha256':{p:sha(p) for p in plan['sha256']},'git_diff_check':'passed','evaluation_scoring_changed':False})
    print(conclusion,flush=True)
if __name__=='__main__':main()
