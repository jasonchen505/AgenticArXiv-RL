#!/usr/bin/env python3
"""New paper assets and gold paths, without learning or model inference."""
from pathlib import Path
import copy,datetime,fcntl,hashlib,json,os,sys,tempfile
R=Path('artifacts/reproduction/round-017-2026-10-06');D=Path('data/repro_round017')
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def state(phase,**kw):write(R/'world-build-status.json',{'phase':phase,'updated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'GPU_jobs':0,'model_inference':False,**kw})
def main():
    plan=read(R/'world-recovery-plan.json')
    for p,h in plan['sha256'].items():assert sha(p)==h,p
    assert not (D/'snapshot.json').exists()
    os.environ.update(STORE_BACKEND='memory',FIGURE_ANALYSIS_BACKEND='extractive_caption',CUDA_VISIBLE_DEVICES='')
    for key,name in [('PDF_RAW_PATH','pdf_raw'),('PDF_FIGURES_PATH','pdf_figures'),('PDF_TRANSLATED_PATH','pdf_translated')]:os.environ[key]=str((D/name).resolve())
    sys.path[:0]=['AgenticArxiv','scripts']
    from rl.env import MockArxivEnv
    from rl.build_snapshot import _unique_snapshot_papers,_prefetch_pdfs,_snapshot_paper_content
    from tools.bootstrap import require_all_tools
    require_all_tools('Round017 independent paperworld')
    with (R/'world-build.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env=MockArxivEnv(snapshot_path=D/'snapshot.json',mode='record',offline_download=False)
        bank=read(R/'paper-bank-admitted.json');assert bank['overlap_with_prior_world_base_ids']==0 and len(bank['pools'])==10
        for pool in bank['pools']:
            args=pool['args'];env.snapshot.setdefault(pool['tool'],{})[json.dumps(args,sort_keys=True)]={'args':args,'result':pool['papers']}
        original_pools=copy.deepcopy({k:env.snapshot[k] for k in ['get_recently_submitted_cs_papers','search_arxiv_papers']})
        for tool,pools in original_pools.items():
            for key,entry in pools.items():env.snapshot[tool][key]['result']=entry['result'][:10 if entry['args'].get('aspect')=='AI' else 5]
        papers=_unique_snapshot_papers(env)
        state('prefetching_PDFs',required_papers=len(papers))
        fetched,cached,failed=_prefetch_pdfs(papers,4,budget_seconds=300)
        state('extracting_caption_world',required_papers=len(papers),prefetch_downloaded=fetched,prefetch_cached=cached,prefetch_failed=failed)
        ok,fail=_snapshot_paper_content(env);assert fail==0 and ok==len(papers),(ok,fail)
        for tool,pools in original_pools.items():env.snapshot[tool]=pools
        ai=next(p['result'] for p in env.snapshot['get_recently_submitted_cs_papers'].values() if p['args']['aspect']=='AI')
        figures={e['result']['paper_id']:e['result']['count'] for e in env.snapshot['extract_paper_figures'].values()}
        swaps=[]
        if figures.get(ai[4]['id'],0)<1:
            index=next(i for i,p in enumerate(ai[:10]) if figures.get(p['id'],0)>=1)
            before=[p['id'] for p in ai];ai[4],ai[index]=ai[index],ai[4];swaps.append({'aspect':'AI','swap_one_based':[5,index+1],'before':before,'after':[p['id'] for p in ai],'reason':'PreregisteredAIref5figureavailability,no modeloutput'})
        assert figures.get(ai[4]['id'],0)>=1
        env.save_snapshot();write(R/'resource-curation.json',{'swaps':swaps,'model_output_used':False,'content_papers':len(papers),'content_ok':ok,'content_failed':0,'figure_backend':'extractive_caption'})
        tasks=read(R.parent/'round-014-2026-10-05/fresh-tasks.json');split=read(R.parent/'round-014-2026-10-05/fresh-splits.json');new_groups={k:[] for k in split['split']}
        cv=next(p['result'] for p in env.snapshot['get_recently_submitted_cs_papers'].values() if p['args']['aspect']=='CV')
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
        files=[D/'snapshot.json',R/'world-tasks.json',R/'world-oracles.json',R/'world-splits.json',R/'paper-bank.json',R/'resource-curation.json',R/'world-gold-validation.json']+[p for p in D.rglob('*') if p.is_file() and p.name!='snapshot.json']
        write(R/'world-data-manifest.json',{'paper_world_disjoint_base_ids':True,'metadata_unique_papers':bank['unique_papers'],'content_papers':len(papers),'gold_passed':22,'tasks':22,'observed_programs':True,'authorVLM_reproduced':False,'sha256':{str(p):sha(p) for p in files}})
        state('world_ready',metadata_unique_papers=bank['unique_papers'],content_papers=len(papers),gold_passed=22,model_evaluation_launched=False)
if __name__=='__main__':
    try:main()
    except Exception as exc:state('failed',error=str(exc));raise
