#!/usr/bin/env python3
"""CPU-only train-world admission, unchanged expert rendering, equal-row controls."""
from pathlib import Path
import collections,copy,datetime,hashlib,json,os,re,sys,tempfile
R=Path('artifacts/reproduction/round-022-2026-10-07');D=Path('data/repro_round022');OLD=R.parent/'round-021-2026-10-06'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def state(phase,**kw):write(R/'train-data-status.json',dict(phase=phase,updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),GPU_jobs=0,new_training_steps=0,**kw))
def main():
    R.mkdir(parents=True,exist_ok=True);D.mkdir(parents=True,exist_ok=True)
    assert not (R/'train-data-plan.json').exists()
    source=Path('data/repro_round014/certified_combined.jsonl');taskpath=R.parent/'round-013-2026-10-05/collection-oracles.json'
    splitpath=Path('data/splits/v3_81.json');tasks=read(taskpath);rows=[json.loads(s) for s in source.read_text().splitlines()]
    experts=[x for x in rows if x['pair_type']=='expert_replay'];errors=[x for x in rows if x['pair_type']!='expert_replay']
    split=read(splitpath)['split'];train=set(split['train']);held=set(split['dev']+split['iid_test']+split['ood_test'])
    assert len(tasks)==137 and len(experts)==398 and len(errors)==45 and len(train)==51 and not train&held
    assert {t['parent_task_id'] for t in tasks}==train=={x['parent_task_id'] for x in experts}
    sources=[source,taskpath,splitpath,Path('data/repro_20261002/snapshot_curated_v1.json'),Path('scripts/generate_sft_data.py'),Path('AgenticArxiv/rl/env.py'),Path('AgenticArxiv/benchmark/semantic_oracle.py'),Path('AgenticArxiv/benchmark/metrics.py'),Path('AgenticArxiv/agents/prompt_templates.py'),Path(__file__),OLD/'next-mainline-plan.json']
    for w in ['trainA','trainB']:
        manifest=read(OLD/(w+'-asset-manifest.json'));assert manifest['phase']=='assets_ready_not_gold'
        for p,h in manifest['sha256'].items():assert sha(p)==h,p
        sources += [OLD/(w+'-asset-manifest.json'),OLD/(w+'-paper-bank.json'),Path('data/repro_round021')/w/'snapshot.json']
    write(R/'train-data-plan.json',{'registered_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_only':True,'GPU_jobs':0,'new_training_steps':0,'train_parents':51,'semantics_each_world':137,'expert_decisions_each_world':398,'equal_rows_each':1239,'steps_each_future':310,'source_and_45_error_rows_unchanged':True,'teacher_rendering':'Unchanged generate_sft_data count-only list observations and str(result)[:500]; exact original tools-description order and expert Thought strings. Not closed observation intervention.','resource_admission_rule':'Derive only train-program ordinal slot requirements for nonempty requested sections and figure index availability. For each existing10-paper aspect pool choose lexicographically first permutation of original candidate indices satisfying all slots, preserving membership. Stop if no perfect assignment; no replacement, fabrication or model scores. Snapshot copied, originals unchanged.','entity_retargeting_rule':'Only5 train positive title-fragment tasks: bind same original search-pool target ordinal. Choose first unique contiguous title token fragment with same word count as original, or first unique longer fragment. Replace only that literal in task/action. Prefix ambiguity not guaranteed; lexical hardness changes disclosed. Preserve all numeric/null refs, program/order/other args, blocked reasons,137semantics,51parents. Remove obsolete old-world diagnostic notes; recompute semantic expected IDs through unchanged oracle.','training_gate':'All137gold in eachworld,398rows each,all1239source/split/target/token audit before admission. Scratch env paths set before imports; real assets hash unchanged after gold. No new learning until full numeric protocol and >=2fresh confirmation worlds/tasks frozen.','sha256':{str(p):sha(p) for p in sources}})
    os.environ.update(STORE_BACKEND='memory',CUDA_VISIBLE_DEVICES='',FIGURE_ANALYSIS_BACKEND='extractive')
    with tempfile.TemporaryDirectory(prefix='round022-gold-') as tmp:
        for key,name in [('PDF_RAW_PATH','raw'),('PDF_TRANSLATED_PATH','translated'),('PDF_FIGURES_PATH','figures')]:os.environ[key]=str(Path(tmp)/name)
        sys.path[:0]=['AgenticArxiv','scripts']
        from tools.bootstrap import require_all_tools
        require_all_tools('Round022 train-only world admission')
        from rl.env import MockArxivEnv
        from models.store import use_memory_store
        from agents.side_effects import LocalSideEffectManager
        from agents.prompt_templates import get_react_prompt,build_visible_setup_context
        from generate_sft_data import execute_expert_tool,build_sft_samples_from_history
        from benchmark.semantic_oracle import attach_expected_paper_ids
        from benchmark.metrics import extract_metrics,is_strict_success
        from transformers import AutoTokenizer
        desc=experts[0]['prompt'][0]['content'].split('你有以下工具可以使用：\n\n',1)[1].split('\n\n当前任务：',1)[0]
        grouped=collections.defaultdict(list)
        for x in experts:grouped[x['source_task_id']].append(x)
        requirements=collections.defaultdict(lambda:collections.defaultdict(lambda:{'sections':set(),'figures':0}))
        for t in tasks:
            aspect=None
            for a in t.get('setup',[])+[{'name':n,'args':a} for n,a in zip(t['expected_tools'],t['expected_tool_args'])]:
                n,args=a['name'],a.get('args',{})
                if n=='get_recently_submitted_cs_papers':aspect=args['aspect']
                if n in ['get_paper_content','analyze_figure']:
                    ref=args['ref'];assert isinstance(ref,int) and aspect
                    if n=='get_paper_content' and args.get('section'):requirements[aspect][ref]['sections'].add(args['section'])
                    if n=='analyze_figure':requirements[aspect][ref]['figures']=max(requirements[aspect][ref]['figures'],args.get('figure_no',1))
        allnew=[];audits={};bindings=[]
        for w in ['trainA','trainB']:
            state('curating_train_resource_slots',world=w)
            originalpath=Path('data/repro_round021')/w/'snapshot.json';snapshot=read(originalpath);bank=read(OLD/(w+'-paper-bank.json'));permutations=[]
            sections=collections.defaultdict(set);figures={}
            for e in snapshot['get_paper_content'].values():
                r=e['result']
                if r.get('content','').strip() and r.get('section'):sections[r['paper_id']].add(r['section'])
            for e in snapshot['extract_paper_figures'].values():figures[e['result']['paper_id']]=e['result']['count']
            for key,e in snapshot['get_recently_submitted_cs_papers'].items():
                aspect=e['args']['aspect'];papers=e['result'];assert len(papers)==10
                def feasible(slot,i):
                    req=requirements[aspect].get(slot,{'sections':set(),'figures':0});pid=papers[i]['id']
                    return req['sections']<=sections[pid] and figures.get(pid,0)>=req['figures']
                def assign(chosen):
                    if len(chosen)==10:return chosen
                    for i in range(10):
                        if i not in chosen and feasible(len(chosen)+1,i):
                            ans=assign(chosen+[i])
                            if ans is not None:return ans
                perm=assign([]);assert perm is not None,('resource slot assignment',w,aspect)
                e['result']=[papers[i] for i in perm]
                permutations.append({'aspect':aspect,'original_indices_one_based':[i+1 for i in perm],'before':[p['id'] for p in papers],'after':[p['id'] for p in e['result']]})
            target=D/(w+'-snapshot.json');write(target,snapshot);target_sha=sha(target)
            write(R/(w+'-resource-curation.json'),{'rule':'first original-index lexicographic feasible permutation','only_train_requirements':True,'used_model_outputs':False,'requirements':{k:{str(i):{'sections':sorted(v['sections']),'figures':v['figures']} for i,v in slots.items()} for k,slots in requirements.items()},'permutations':permutations,'original_snapshot_sha256':sha(originalpath),'curated_snapshot_sha256':sha(target)})
            transformed=[]
            for original in tasks:
                t=copy.deepcopy(original);t.pop('expected_paper_ids',None);t.pop('note',None)
                for step,args in enumerate(t['expected_tool_args']):
                    oldref=args.get('ref')
                    if not isinstance(oldref,str):continue
                    use_memory_store(reset=True);oldenv=MockArxivEnv('data/repro_20261002/snapshot_curated_v1.json',mode='replay');setup=t['setup'][0]
                    oldpapers=oldenv.execute_tool(setup['name'],setup['args']);oldid=original['expected_paper_ids'][step];index=next(i for i,p in enumerate(oldpapers) if p['id']==oldid)
                    newenv=MockArxivEnv(target,mode='replay');newpapers=newenv.execute_tool(setup['name'],setup['args']);title=newpapers[index]['title'];tokens=title.split();size=len(oldref.split());fragment=None
                    for length in range(size,len(tokens)+1):
                        for start in range(len(tokens)-length+1):
                            candidate=' '.join(tokens[start:start+length])
                            if sum(candidate.lower() in p['title'].lower() for p in newpapers)==1:fragment=candidate;break
                        if fragment is not None:break
                    assert fragment is not None and oldref in t['task']
                    t['task']=t['task'].replace(oldref,fragment);args['ref']=fragment
                    bindings.append({'world':w,'task_id':t['id'],'original_literal':oldref,'new_literal':fragment,'target_ordinal':index+1,'target_paper_id':newpapers[index]['id'],'old_word_count':size,'new_word_count':len(fragment.split()),'prefix_ambiguity_preserved':False})
                transformed.append(t)
            write(R/(w+'-tasks.json'),transformed)
            state('CPU_gold_and_expert_generation',world=w)
            oracles=attach_expected_paper_ids(transformed,target);write(R/(w+'-oracles.json'),oracles)
            env=MockArxivEnv(target,mode='replay',offline_download=True);newby={};gold=[];tracefile=(R/(w+'-gold-traces.jsonl')).open('w')
            for t in oracles:
                use_memory_store(reset=True);env.reset_runtime_state();side=LocalSideEffectManager();side._translate_seq=0;session='sft_spec_'+t['id'];history=[]
                for a in t.get('setup',[]):execute_expert_tool(env,side,a['name'],a.get('args',{}),session)
                sequence=sorted(grouped[t['id']],key=lambda x:x['trajectory_step']);assert len(sequence)==len(t['expected_tools'])+1
                for i,x in enumerate(sequence):
                    oldchosen=x['chosen'][0]['content'];thought=oldchosen.split('\nAction: ',1)[0].removeprefix('Thought: ')
                    if i<len(t['expected_tools']):
                        name,args=t['expected_tools'][i],t['expected_tool_args'][i];result=execute_expert_tool(env,side,name,args,session)
                        if isinstance(result,dict) and name not in ['get_recently_submitted_cs_papers','search_arxiv_papers']:assert str(result.get('paper_id'))==t['expected_paper_ids'][i]
                        obs=f'成功获取 {len(result)} 篇论文' if isinstance(result,list) else str(result)[:500]
                        action=json.dumps({'name':name,'args':args},ensure_ascii=False)
                    else:action='FINISH';obs='任务完成'
                    history.append({'thought':thought,'action':action,'observation':obs})
                metrics=extract_metrics(t,{'history':history},'cpu_train_world_gold',0,session);assert is_strict_success(metrics),(w,t['id'],metrics.to_dict())
                generated=build_sft_samples_from_history(t['task'],desc,history,source_task_id=t['id'],source_split='train',initial_history=build_visible_setup_context(t))
                assert len(generated)==len(sequence)
                for x,g in zip(sequence,generated):
                    nx=copy.deepcopy(x);nx['prompt']=[g['messages'][0]];nx['chosen']=[g['messages'][1]];nx['rejected']=copy.deepcopy(nx['chosen']);nx['training_world']=w;nx['world_data_provenance']={'snapshot_sha256':target_sha,'original_semantic_task_id':t['id']}
                    newby[(t['id'],x['trajectory_step'])]=nx
                gold.append({'task_id':t['id'],'parent_task_id':t['parent_task_id'],'strict_success':True,'decisions':len(sequence)})
                tracefile.write(json.dumps({'task_id':t['id'],'history':history,'metrics':metrics.to_dict()},ensure_ascii=False)+'\n')
            tracefile.close();ordered=[newby[(x['source_task_id'],x['trajectory_step'])] for x in experts];assert len(ordered)==398
            write(R/(w+'-gold-validation.json'),{'tasks':137,'passed':137,'train_parents':51,'heldout_parent_overlap':0,'used_model_inference':False,'results':gold,'env_stats':env.stats})
            (D/(w+'-experts.jsonl')).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in ordered));allnew.append(ordered)
            for p,h in read(OLD/(w+'-asset-manifest.json'))['sha256'].items():assert sha(p)==h,('asset modified by gold',p)
            audits[w]={'gold_passed':137,'expert_rows':398,'chosen_changed_rows':sum(x['chosen']!=y['chosen'] for x,y in zip(experts,ordered)),'prompt_changed_rows':sum(x['prompt']!=y['prompt'] for x,y in zip(experts,ordered)),'real_asset_hashes_unchanged':True,'curated_snapshot_sha256':sha(target)}
        write(R/'entity-bindings.json',{'count':len(bindings),'bindings':bindings,'lexical_hardness_not_preserved':True,'heldout_entity_patterns_used':False})
        world=experts+allnew[0]+allnew[1]+errors;repeat=experts*3+errors;assert len(world)==len(repeat)==1239
        assert world[-45:]==repeat[-45:]==errors and world[:398]==repeat[:398]==experts
        tokenizer=AutoTokenizer.from_pretrained('outputs/repro_20261003/sft_e1/final',local_files_only=True);lengths={};files=[]
        for name,data in [('world',world),('repeat',repeat)]:
            token_lengths=[]
            for x in data:
                assert x['parent_task_id'] in train and x['parent_task_id'] not in held
                assert x['pair_type']!='expert_replay' or x['chosen']==x['rejected']
                for key in ['chosen','rejected']:
                    ids=tokenizer.apply_chat_template(x['prompt']+x[key],tokenize=True)
                    if hasattr(ids,'keys'):ids=ids['input_ids']
                    token_lengths.append(len(ids))
            assert max(token_lengths)<4096,(name,max(token_lengths))
            lengths[name]={'maximum':max(token_lengths),'mean':sum(token_lengths)/len(token_lengths),'sequences':len(token_lengths),'truncation':0}
            p=D/(name+'_combined.jsonl');p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in data));files.append(p)
        for p,h in read(R/'train-data-plan.json')['sha256'].items():assert sha(p)==h,('frozen input changed',p)
        result={'phase':'data_admitted_no_learning','worlds':audits,'rows_each':1239,'expert_rows_each':1194,'original_error_rows_each_unchanged':45,'train_parents':51,'heldout_parent_overlap':0,'gold_passed_total':274,'source_semantic_set_unchanged':True,'teacher_format_unchanged':True,'original_description_and_thoughts_preserved':True,'entity_title_bindings':len(bindings),'lexical_hardness_not_preserved':True,'token_lengths':lengths,'same_steps_rows_not_identical_token_compute':True,'learning_steps_each_planned':310,'GPU_jobs':0,'new_training_steps':0,'learning_permitted':False,'remaining_gate':'Full numerical scientific protocol plus >=2fresh confirmation paper worlds/tasks freeze BEFORE any learning.','sha256':{str(p):sha(p) for p in files+[D/'trainA-snapshot.json',D/'trainB-snapshot.json',R/'entity-bindings.json']}}
        write(R/'train-data-audit.json',result);state('data_admitted_no_learning',gold_passed=274,rows_each=1239,learning_permitted=False);print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as exc:state('failed',error=str(exc));raise
