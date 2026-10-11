#!/usr/bin/env python3
"""Close matched-objective experiment and render remote HTML automatically."""
from pathlib import Path
from datetime import datetime
import collections,hashlib,json,subprocess,sys
R=Path('artifacts/reproduction/round-010-2026-10-04');ARMS=['ct43','dpo43','ct44','dpo44'];MODELS=['sft']+ARMS
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def good(d):return d['completed'] and d['tool_accurate'] and d['arg_score']==1 and d['ref_score']==1 and d['parse_fail']==0 and d['tool_fail']==0 and d['terminal_semantics_accurate']
def leaf(a,g):return a+'-dev' if g=='dev' else g.split('-')[0]+'-'+a+'-'+g.split('-',1)[1]
def main():
    assert read(R/'completion.json')['state']=='completed';plan=read(R/'experiment-plan.json')
    for p,h in {**plan['sha256'],**plan['reused_evidence_sha256']}.items():assert sha(p)==h,p
    first_train=min(datetime.fromisoformat(read(R/(a+'-train-job.json'))['started_at_utc']).timestamp() for a in ARMS)
    first_eval=min(datetime.fromisoformat(read(p)['started_at_utc']).timestamp() for p in R.glob('*-job.json') if 'train-job' not in p.name)
    assert (R/'fresh-tasks.json').stat().st_mtime<=(R/'experiment-plan.json').stat().st_mtime<=first_train
    training={};chronology={'first_train_start_unix':first_train,'first_model_eval_start_unix':first_eval,'fresh_tasks_mtime_unix':(R/'fresh-tasks.json').stat().st_mtime,'candidate_freezes':{}}
    for a in ARMS:
        manifest=read(R/(a+'-training-manifest.json'));frozen=read(R/(a+'-candidate-freeze.json'));job=read(R/(a+'-train-job.json'))
        assert job['exit_code']==0 and manifest['actual_steps']==51 and manifest['seed']==int(a[-2:]) and manifest['source_adapter_sha256']==plan['source_adapter_sha256']
        assert manifest['dataset_rows']==448 and manifest['truncation_count']==0 and frozen['changed_tensors']==392
        assert sha(Path(frozen['model'])/'adapter_model.safetensors')==frozen['candidate_sha256']
        if a.startswith('dpo'):assert manifest['reference_equal_sft_and_frozen_tensors']==392 and frozen['reference_equal_sft_tensors']==392
        finished=datetime.fromisoformat(job['finished_at_utc']).timestamp();freezetime=(R/(a+'-candidate-freeze.json')).stat().st_mtime;assert finished<=freezetime<=first_eval
        chronology['candidate_freezes'][a]={'finish_unix':finished,'freeze_mtime_unix':freezetime,'before_all_model_evaluation':True}
        training[a]={**{k:manifest[k] for k in ['objective','seed','actual_steps','dataset_rows','effective_batch','lr','train_loss','trainer_elapsed_seconds','torch_peak_gib','reference_equal_sft_and_frozen_tensors']},'process_seconds':job['elapsed_seconds'],'sampled_device_max_mib':job['sampled_max_device_used_mib'],'changed_tensors':392,'candidate_sha256':frozen['candidate_sha256']}
    write(R/'training-audit.json',training);write(R/'freeze-chronology-audit.json',chronology)
    metrics={a:{} for a in MODELS};task_counts={};new_count=old_count=0
    pools={'dev':read('data/splits/v3_81.json')['split']['dev']}
    for prefix,path in [('previous',Path('artifacts/reproduction/round-008-2026-10-04')),('fresh',R)]:pools.update({prefix+'-'+g:ids for g,ids in read(path/'fresh-splits.json')['split'].items()})
    for group,ids in pools.items():
        for a in MODELS:
            directory=R/leaf(a,group);summary=read(directory/'summary.json');details=summary['details'];assert summary['error_count']==0 and summary['sample_count']==len(ids)*3
            assert {d['task_id'] for d in details}==set(ids) and len({(d['task_id'],d['trial']) for d in details})==len(ids)*3
            if a=='sft' and (group=='dev' or group.startswith('previous')):old_count+=len(details)
            else:new_count+=len(details);assert read(R/(directory.name+'-job.json'))['exit_code']==0
            comp=read(R/(('sft-vs-ct43-' if a=='sft' else 'sft-vs-'+a+'-')+group+'.json'))['models'][a];assert comp['authoritative_rescore_drift']==0
            successes=sum(good(d) for d in details);assert successes==round(comp['strict_success_rate']*comp['runs'])
            metrics[a][group]={**comp,'strict_successes':successes,'pass3_tasks':round(comp['reliability']['pass^3']*comp['tasks']),'false_finish_count':sum(d['false_finish'] for d in details)}
            per=collections.Counter()
            for d in details:per[d['task_id']]+=int(good(d))
            task_counts.setdefault(group,{})[a]=dict(per)
        for seed in [43,44]:assert all(x['authoritative_rescore_drift']==0 for x in read(R/(f'ct{seed}-vs-dpo{seed}-'+group+'.json'))['models'].values())
    assert new_count==690 and old_count==90
    totals={prefix:{a:sum(metrics[a][prefix+'-'+s]['strict_successes'] for s in ['iid_test','ood_test']) for a in MODELS} for prefix in ['previous','fresh']}
    effects={prefix:{str(s):totals[prefix]['dpo'+str(s)]-totals[prefix]['ct'+str(s)] for s in [43,44]} for prefix in totals}
    fresh=totals['fresh'];stable=all(effects['fresh'][str(s)]>0 for s in [43,44]);above=all(fresh['dpo'+str(s)]>fresh['sft'] for s in [43,44])
    conclusion=f"新22任务：SFT {fresh['sft']}/66；继续SFT seed43 {fresh['ct43']}/66、seed44 {fresh['ct44']}/66；DPO seed43 {fresh['dpo43']}/66、seed44 {fresh['dpo44']}/66。"
    conclusion+=('DPO在两个种子均高于继续SFT。' if stable else 'DPO没有在两个种子均高于继续SFT。')
    conclusion+=('两个DPO均高于原SFT，仍需更大独立确认。' if above else '未建立两个DPO种子均超过原SFT的收益，保留原SFT参考。')
    write(R/'mainline-outcome.json',{'plain_chinese':conclusion,'metrics':metrics,'strict_totals':totals,'dpo_minus_continued_sft':effects,'both_fresh_dpo_seeds_gain_vs_continuation':stable,'both_fresh_dpo_seeds_above_source_sft':above,'reference_model':plan['evaluation_models']['sft'],'no_automatic_model_promotion':True,'new_runs':new_count,'reused_development_runs':old_count})
    write(R/'task-level-audit.json',{'strict_counts':task_counts})
    changed=[]
    for group in pools:
        for t in sorted(task_counts[group]['sft']):
            counts={a:task_counts[group][a][t] for a in MODELS}
            ff={};raw={}
            for a in MODELS:
                ff[a]=sum(d['false_finish'] for d in read(R/leaf(a,group)/'summary.json')['details'] if d['task_id']==t)
            if len(set(counts.values()))==1 and len(set(ff.values()))==1:continue
            for a in MODELS:raw[a]=[json.loads(s) for s in (R/leaf(a,group)/'traces.jsonl').read_text().splitlines() if json.loads(s)['task_id']==t]
            changed.append({'group':group,'task_id':t,'strict_counts':counts,'premature_finish_counts':ff,'raw_trials':raw})
    write(R/'behavior-audit.json',{'changed_tasks':changed,'premature_finish_definition':'Repository false_finish includes FINISH with missing required tools and may be an explicit refusal, not a verbal claim of completion.'})
    completion=read(R/'completion.json');completion.update(actual_new_benchmark_runs=new_count,actual_reused_development_runs=old_count,actual_comparison_optimizer_steps=204,diagnostic_attempt_optimizer_steps=51,planned_comparison_training_prompt_exposures=816);write(R/'completion.json',completion)
    doc=f"# 第010轮：DPO与继续SFT的两个种子对照\n\n{conclusion}\n\n同一sourceSFT、448prompt/chosen、51步有效batch4、lr5e-6，DPO额外使用合成负例和冻结SFT参考（392张量等值）。只读训练集，137语义来自51训练父任务，388正常错误拒绝+60阻塞错误回答，未读测试轨迹。\n\n690次新评测，复用90次开发记录；无异常或权威严格重计分漂移。新请求22条标准路径通过且在输出前冻结，同一已知论文世界与观察过的留出父任务，不是新世界泛化；原OOD标签不保证新工具程序。\n\n仅GPU4，GPU5–7不用。最终step51权重统一冻结后评测，无测试驱动调参或最佳种子选取。对照优化步相同，DPO与SFT计算量不同。\n"
    Path('docs/reproduction_round_010.md').write_text(doc);(R/'reproduction_round_010.md').write_text(doc)
    subprocess.run(['git','diff','--check'],check=True)
    write(R/'final-source-manifest.json',{'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'frozen_source_sha256':{p:sha(p) for p in plan['sha256']},'git_diff_check':'passed','evaluation_scoring_changed':False})
    print(conclusion,flush=True)
    if Path('scripts/build_round010_report.py').exists():subprocess.run(['.venv-repro/bin/python','scripts/build_round010_report.py'],check=True)
if __name__=='__main__':main()
