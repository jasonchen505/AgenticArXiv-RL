#!/usr/bin/env python3
"""Complete all recorded T4 × enum T5 observations and validate the unchanged world."""
from pathlib import Path
import copy,datetime,fcntl,hashlib,json,os,sys
R=Path('artifacts/reproduction/round-017-2026-10-06'); A=Path('artifacts/reproduction/round-018-2026-10-06'); D=Path('data/repro_round017')
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def state(phase,**kw):write(R/'world-build-status.json',{'phase':phase,'updated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'GPU_jobs':0,'model_inference':False,**kw})

def main():
    assert not (R/'cross-world-experiment-plan.json').exists()
    assert sha(D/'snapshot.json')==read(A/'data-repair-audit.json')['after_snapshot_sha256']
    assets=[p for p in D.rglob('*') if p.is_file()]
    frozen={str(p):sha(p) for p in assets}
    write(A/'gold-resumption-plan.json',{'registered_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_only':True,'GPU_jobs':0,'reason':'Existing frozen config requires scratch paths supplied before imports; original assets restored and excluded from rollout sandbox.','sha256':{**frozen,str(Path(__file__)):sha(Path(__file__)),str(A/'data-repair-audit.json'):sha(A/'data-repair-audit.json')}})
    os.environ.update(STORE_BACKEND='memory',FIGURE_ANALYSIS_BACKEND='extractive',CUDA_VISIBLE_DEVICES='')
    for key,name in [('PDF_RAW_PATH','pdf_raw'),('PDF_FIGURES_PATH','pdf_figures'),('PDF_TRANSLATED_PATH','pdf_translated')]:os.environ[key]=str((A/'gold_scratch'/name).resolve())
    sys.path[:0]=['AgenticArxiv','scripts']
    from tools.bootstrap import require_all_tools
    require_all_tools('Round018 isolated CPU gold')
    snapshot=read(D/'snapshot.json')
    bank=read(R/'paper-bank-admitted.json'); assert bank['overlap_with_prior_world_base_ids']==0
    ai=next(p['result'] for p in snapshot['get_recently_submitted_cs_papers'].values() if p['args']['aspect']=='AI')
    cv=next(p['result'] for p in snapshot['get_recently_submitted_cs_papers'].values() if p['args']['aspect']=='CV')
    papers=list(snapshot['extract_paper_figures'])

    tasks=read(R.parent/'round-014-2026-10-05/fresh-tasks.json');split=read(R.parent/'round-014-2026-10-05/fresh-splits.json');new_groups={k:[] for k in split['split']}
    for t in tasks:
        old=t['id'];t['id']=old.replace('fresh014_','world017_')
        for g,ids in split['split'].items():
            if old in ids:new_groups[g].append(t['id'])
        if t['parent_task_id']=='ref_stress_learning_state':
            title=ai[2]['title'];t['task']='下载当前列表中标题为“'+title+'”的论文。';t['expected_tool_args']=[{'ref':title}]
        if t['parent_task_id']=='ref_ctrl_id_cache':
            paper_id=cv[0]['id'];t['task']='请检查论文'+paper_id+'的缓存状态。';t['expected_tool_args']=[{'ref':paper_id}]
    from benchmark.semantic_oracle import attach_expected_paper_ids,_call
    from benchmark.task_spec import reference_terminal_thought
    from benchmark.metrics import is_strict_success
    from rl.multiturn_env import AgenticArxivMultiTurnEnv
    from rl.reward import RewardCalculator
    reward=RewardCalculator(curriculum_steps=0);oracles=[];results=[];traces=[]
    state('CPU_gold_validation',tasks=22)
    for t in tasks:
        t=attach_expected_paper_ids([t],D/'snapshot.json')[0];world=AgenticArxivMultiTurnEnv(D/'snapshot.json');world.reset(task_id=t['id']);history=[]
        for action in t.get('setup',[]):_call(world,action['name'],action.get('args',{}))
        for name,args in zip(t['expected_tools'],t['expected_tool_args']):
            obs=_call(world,name,args)
            if isinstance(obs,list):assert not any(isinstance(p,dict) and p.get('_mock_env',{}).get('offline_fallback') for p in obs)
            history.append({'thought':'执行预声明工具步骤','action':json.dumps({'name':name,'args':args},ensure_ascii=False),'observation':json.dumps(obs,ensure_ascii=False,default=str)})
        history.append({'thought':reference_terminal_thought(t),'action':'FINISH','observation':'任务完成'})
        breakdown,m=reward.compute_reward_breakdown(t,{'history':history});passed=is_strict_success(m) and breakdown.total>=.99
        results.append({'task_id':t['id'],'passed':passed});oracles.append(t);traces.append({'task_id':t['id'],'history':history})
    assert len(results)==22 and all(x['passed'] for x in results),results
    write(R/'world-tasks.json',tasks);write(R/'world-oracles.json',oracles);write(R/'world-splits.json',{'split':new_groups});write(R/'world-gold-validation.json',{'tasks':22,'passed':22,'model_inference':False,'results':results})
    (R/'world-gold-traces.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in traces))
    for p,h in frozen.items():assert sha(p)==h,p
    files=[A/'gold-resumption-plan.json',A/'data-repair-plan-v2.json',A/'data-repair-audit.json',A/'snapshot-before-T5-repair.json',R/'paper-bank-admitted.json',R/'world-gold-traces.jsonl',D/'snapshot.json',R/'world-tasks.json',R/'world-oracles.json',R/'world-splits.json',R/'paper-bank.json',R/'resource-curation.json',R/'world-gold-validation.json']+[p for p in D.rglob('*') if p.is_file() and p.name!='snapshot.json']
    write(R/'world-data-manifest.json',{'paper_world_disjoint_base_ids':True,'metadata_unique_papers':bank['unique_papers'],'content_papers':len(papers),'gold_passed':22,'tasks':22,'observed_programs':True,'authorVLM_reproduced':False,'sha256':{str(p):sha(p) for p in files}})
    state('world_ready',metadata_unique_papers=bank['unique_papers'],content_papers=len(papers),gold_passed=22,model_evaluation_launched=False)
if __name__=='__main__':
    try:main()
    except Exception as exc:state('failed',error=str(exc));raise
