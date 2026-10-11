#!/usr/bin/env python3
"""CPU confirmation recovery; original failed paper retained with genuine T2 error."""
from pathlib import Path
import argparse,datetime,fcntl,hashlib,json,os,subprocess,sys,time
R=Path('artifacts/reproduction/round-022-2026-10-07');A=R.parent/'round-023-2026-10-07'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
write=lambda p,x:Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def state(phase,**kw):write(A/'confirmation-recovery-status.json',dict(phase=phase,updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),GPU_jobs=0,new_training_steps=0,**kw))
def child(world):
    import functools
    D=Path('data/repro_round022_confirmation')/world;D.mkdir(parents=True,exist_ok=True);assert not (D/'snapshot.json').exists()
    bank=read(R/(world+'-paper-bank.json'));os.environ.update(STORE_BACKEND='memory',FIGURE_ANALYSIS_BACKEND='extractive',CUDA_VISIBLE_DEVICES='')
    for key,name in [('PDF_RAW_PATH','raw'),('PDF_TRANSLATED_PATH','translated'),('PDF_FIGURES_PATH','figures')]:os.environ[key]=str((D/name).resolve())
    sys.path.insert(0,'AgenticArxiv')
    from tools.bootstrap import require_all_tools
    require_all_tools('Round023 CPU asset recovery')
    from rl.env import MockArxivEnv
    from rl import build_snapshot as builder
    from rl.backfill_figure_analysis import backfill_entries
    from tools import paper_content_tool as content,paper_summary_tool as summary
    from models.schemas import Paper
    from models.store import store
    env=MockArxivEnv(D/'snapshot.json',mode='record',offline_download=False)
    for pool in bank['pools']:
        args=pool['args'];key=env._keyword_search_key(args) if pool['tool']=='search_arxiv_papers' else env._make_key(args)
        env.snapshot.setdefault(pool['tool'],{})[key]={'args':args,'result':pool['papers']}
    papers=builder._unique_snapshot_papers(env);assert len(papers)==bank['unique_base_ids'];fetched,cached,failed=builder._prefetch_pdfs(papers,4,budget_seconds=300);assert failed==0
    pdf_sha={str(p.resolve()):sha(p) for p in (D/'raw').glob('*.pdf')};assert len(pdf_sha)==len(papers)
    original_text=content._pdf_to_text;verified=[]
    @functools.lru_cache(maxsize=256)
    def cached_text(path):
        assert sha(path)==pdf_sha[str(Path(path).resolve())]
        one=original_text(path);two=original_text(path);assert one==two
        verified.append({'path':str(path),'text_sha256':hashlib.sha256(one.encode()).hexdigest()});return one
    # Scoped to this CPU process, after metadata/PDF freeze. No core source patch.
    content._pdf_to_text=cached_text;summary._pdf_to_text=cached_text;builder.FIGURE_QUESTIONS=()
    ok,fail=builder._snapshot_paper_content(env)
    default_ids={e['result']['paper_id'] for e in env.snapshot.get('get_paper_content',{}).values() if e['result'].get('section') is None and e['result'].get('content','').strip()}
    unavailable=[]
    for pid in sorted(set(papers)-default_ids):
        store.set_last_papers('__round023_missing__',[Paper(**{k:v for k,v in papers[pid].items() if not k.startswith('_')})]);env.execute_tool('download_arxiv_pdf',{'session_id':'__round023_missing__','ref':1})
        try:env.execute_tool('get_paper_content',{'session_id':'__round023_missing__','ref':1})
        except RuntimeError as e:
            assert "section 'abstract' was not found" in str(e),str(e)
            unavailable.append({'paper_id':pid,'tool':'get_paper_content','section':None,'real_tool_error':str(e),'metadata_retained':True,'fake_observation':False})
        else:raise AssertionError('Missing recorded default unexpectedly succeeded')
        for section in ['abstract','method','result','conclusion']:
            try:env.execute_tool('get_paper_content',{'session_id':'__round023_missing__','ref':1,'section':section})
            except RuntimeError:pass
        for style in builder.SUMMARY_STYLES:
            for words in builder.MAX_WORDS_BUCKETS:
                try:env.execute_tool('summarize_paper',{'session_id':'__round023_missing__','ref':1,'style':style,'max_words':words})
                except RuntimeError:pass
        env.execute_tool('extract_paper_figures',{'session_id':'__round023_missing__','ref':1})
    assert len(unavailable)==fail and ok+fail==len(papers)
    updated,stats=backfill_entries(env.snapshot,backend='extractive');assert stats.failed==0;env.snapshot=updated;env.save_snapshot()
    assert len(env.snapshot['extract_paper_figures'])==len(papers)
    for p,h in pdf_sha.items():assert sha(p)==h,p
    write(A/(world+'-asset-recovery-audit.json'),{'metadata_membership_unchanged':True,'actual_PDFs':len(pdf_sha),'default_content_papers':ok,'default_content_unavailable':unavailable,'real_figures':stats.figures,'T5_entries':stats.added,'cached_text_outputs_verified':len(verified),'cache_hits':cached_text.cache_info().hits,'cache_misses':cached_text.cache_info().misses,'all_original_PDF_hashes_unchanged':True,'tool_core_source_unchanged':True,'GPU_jobs':0,'new_training_steps':0})
    write(R/(world+'-asset-manifest.json'),{'phase':'assets_ready_not_gold','world':world,'metadata_unique_papers':len(papers),'actual_PDFs':len(pdf_sha),'content_papers':ok,'content_unavailable_count':fail,'content_unavailable':unavailable,'figures':stats.figures,'T5_entries':stats.added,'gold_passed':False,'new_training_steps':0,'sha256':{str(p):sha(p) for p in D.rglob('*') if p.is_file()}})
def gold(world):
    import copy,tempfile
    original=Path('data/repro_round022_confirmation')/world/'snapshot.json';snapshot=read(original);permutations=[]
    figures={e['result']['paper_id']:e['result']['count'] for e in snapshot['extract_paper_figures'].values()}
    abstracts={e['result']['paper_id'] for e in snapshot['get_paper_content'].values() if e['result'].get('section')=='abstract' and e['result'].get('content','').strip()}
    for e in snapshot['get_recently_submitted_cs_papers'].values():
        aspect=e['args']['aspect'];papers=e['result']
        if aspect not in ['AI','CR']:continue
        def feasible(slot,i):
            pid=papers[i]['id'];return not (aspect=='AI' and slot==5 and figures.get(pid,0)<1) and not (aspect=='CR' and slot==3 and pid not in abstracts)
        def assign(chosen):
            if len(chosen)==len(papers):return chosen
            for i in range(len(papers)):
                if i not in chosen and feasible(len(chosen)+1,i):
                    ans=assign(chosen+[i])
                    if ans is not None:return ans
        perm=assign([]);assert perm is not None,(world,aspect);e['result']=[papers[i] for i in perm];permutations.append({'aspect':aspect,'permutation_one_based':[i+1 for i in perm],'membership_unchanged':True})
    path=original.parent/'curated-snapshot.json';write(path,snapshot)
    os.environ.update(CUDA_VISIBLE_DEVICES='',STORE_BACKEND='memory',FIGURE_ANALYSIS_BACKEND='extractive')
    with tempfile.TemporaryDirectory(prefix='confirm022-gold-') as tmp:
        for key,name in [('PDF_RAW_PATH','raw'),('PDF_TRANSLATED_PATH','translated'),('PDF_FIGURES_PATH','figures')]:os.environ[key]=str(Path(tmp)/name)
        sys.path[:0]=['AgenticArxiv','scripts']
        from tools.bootstrap import require_all_tools
        require_all_tools('Round022 confirmation CPU gold')
        from benchmark.semantic_oracle import attach_expected_paper_ids
        from benchmark.metrics import extract_metrics,is_strict_success
        from benchmark.task_spec import reference_terminal_thought
        from generate_sft_data import execute_expert_tool
        from agents.side_effects import LocalSideEffectManager
        from models.store import use_memory_store
        from rl.env import MockArxivEnv
        tasks=copy.deepcopy(read(R/'confirmation-task-template.json'));groups=read(R/'confirmation-template-splits.json')['split'];bindings=[]
        for t in tasks:
            t['id']=t['id'].replace('confirm022_',world+'022_')
            parent=t['parent_task_id']
            if parent=='ref_stress_learning_state':
                ai=next(e['result'] for e in snapshot['get_recently_submitted_cs_papers'].values() if e['args']['aspect']=='AI');ref=ai[2]['title'];t['task']='下载当前列表中标题为“'+ref+'”的论文。';t['expected_tool_args']=[{'ref':ref}];bindings.append({'parent':parent,'ordinal':3,'paper_id':ai[2]['id']})
            if parent=='ref_ctrl_id_cache':
                cv=next(e['result'] for e in snapshot['get_recently_submitted_cs_papers'].values() if e['args']['aspect']=='CV');ref=cv[0]['id'];t['task']='请检查论文'+ref+'的缓存状态。';t['expected_tool_args']=[{'ref':ref}];bindings.append({'parent':parent,'ordinal':1,'paper_id':ref})
        oracles=attach_expected_paper_ids(tasks,path);results=[];traces=[]
        for t in oracles:
            use_memory_store(reset=True);env=MockArxivEnv(path,mode='replay',offline_download=True);side=LocalSideEffectManager();session='confirm_gold_'+t['id'];history=[]
            for a in t.get('setup',[]):execute_expert_tool(env,side,a['name'],a.get('args',{}),session)
            for i,(name,args) in enumerate(zip(t['expected_tools'],t['expected_tool_args'])):
                result=execute_expert_tool(env,side,name,args,session);obs=f'成功获取 {len(result)} 篇论文' if isinstance(result,list) else str(result)[:1000]
                history.append({'thought':'执行预声明工具步骤','action':json.dumps({'name':name,'args':args},ensure_ascii=False),'observation':obs})
            history.append({'thought':reference_terminal_thought(t),'action':'FINISH','observation':'任务完成'})
            metrics=extract_metrics(t,{'history':history},'CPU_confirmation_gold',0,session);assert is_strict_success(metrics),(world,t['id'],metrics.to_dict());results.append({'task_id':t['id'],'strict_success':True});traces.append({'task_id':t['id'],'history':history,'metrics':metrics.to_dict()})
        split=read('data/splits/v3_81.json')['split'];assert not {t['parent_task_id'] for t in tasks}&set(split['train']);assert len(results)==22
        write(R/(world+'-tasks.json'),tasks);write(R/(world+'-oracles.json'),oracles);write(R/(world+'-splits.json'),{'split':{k:[x.replace('confirm022_',world+'022_') for x in v] for k,v in groups.items()}})
        write(R/(world+'-gold-validation.json'),{'tasks':22,'passed':22,'train_parent_overlap':0,'resource_permutations':permutations,'entity_bindings':bindings,'new_task_programs':False,'model_outputs':0,'results':results});(R/(world+'-gold-traces.jsonl')).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in traces))
    for p,h in read(R/(world+'-asset-manifest.json'))['sha256'].items():assert sha(p)==h,('original asset changed',p)
    files=[path,R/(world+'-tasks.json'),R/(world+'-oracles.json'),R/(world+'-splits.json'),R/(world+'-gold-validation.json')]
    write(R/(world+'-world-manifest.json'),{'phase':'confirmation_world_ready','gold_passed':22,'model_outputs':0,'original_assets_unchanged':True,'sha256':{str(p):sha(p) for p in files}})

def main():
    import shutil
    A.mkdir(exist_ok=True);assert not (A/'confirmation-recovery-plan-v2.json').exists()
    shutil.copy2(A/'confirmation-recovery-status.json',A/'v1-recovery-failure.json')
    for n in ['confirmation-assets-status.json','confirmation-assets-controller.log','confirmA--world-assets.log']:shutil.copy2(R/n,A/('v2-previous-'+n))
    sources=[A/'confirmation-recovery-plan.json',Path('scripts/recover_round023_confirmation.py'),Path(__file__),R/'confirmation-preparation-plan.json',R/'confirmation-asset-plan.json',Path('scripts/prepare_round022_confirmation_assets.py'),Path('AgenticArxiv/tools/paper_content_tool.py'),Path('AgenticArxiv/tools/paper_summary_tool.py')]+[R/(w+'-paper-bank.json') for w in ['confirmA','confirmB']]
    write(A/'confirmation-recovery-plan-v2.json',{'registered_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_recovery_correction':'Memoization PDF SHA keys canonicalized to absolute resolved paths; retain failed v1 source/plan/status. No tool/model/rule changes.','reason':'One confirmA ROordinal2 realPDF has no extractableAbstract; previous blanketdefault-content gate failed. Retain exact142metadata membership and realPDF requirement, retain genuine absent-Abstract errors, require ALL predeclared44gold. No paper/task/parameter/model/rule replacement.','gate_amendment':'Defaultcontent coverage is disclosed, not fabricated. Only precise unchanged realtool missing-Abstract errors admitted for nonrequired surface; every requested gold operation must still succeed. Other errors stop. RealT4/T5 completed even for default-unavailable papers.','CPU_cache':'Process-only memoization of immutable PDF->text; every firstmiss double-runs original and verifies equal outputs; PDFsha fixed before cache and rechecked after. No learner/scorer/tool-source changes.','GPU_jobs':0,'new_training_steps':0,'sha256':{str(p):sha(p) for p in sources}})
    with (R/'confirmation-assets.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for world in ['confirmA','confirmB']:
            for phase,flag,budget in [('CPU_asset_recovery','--world',1200),('CPU_confirmation_gold','--gold',240)]:
                state(phase,world=world)
                with (A/(world+flag+'.log')).open('w') as log:subprocess.run(['.venv-repro/bin/python','-u',__file__,flag,world],timeout=budget,check=True,stdout=log,stderr=log,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
        state('confirmation_worlds_ready',gold_passed=44,worlds=2,learning_permitted=False)
        write(R/'confirmation-assets-status.json',read(A/'confirmation-recovery-status.json'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--world');p.add_argument('--gold');a=p.parse_args()
    if a.world:child(a.world)
    elif a.gold:gold(a.gold)
    else:
        try:main()
        except Exception as e:state('failed',error=str(e));raise
