#!/usr/bin/env python3
"""Freeze new held-out-parent parameter recombinations, before model output."""
from pathlib import Path
import copy
import hashlib
import json
import os
import sys
import tempfile

R=Path('artifacts/reproduction/round-008-2026-10-04')
SNAPSHOT=Path('data/repro_20261002/snapshot_curated_v1.json')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def signature(t):
    return json.dumps({k:t.get(k,[] if k=='setup' else None) for k in ['expected_tools','expected_tool_args','setup','expected_terminal_mode','expected_terminal_reason']},sort_keys=True,ensure_ascii=False)

def main():
    assert not (R/'fresh-tasks.json').exists(),'Never overwrite a frozen test set'
    R.mkdir(parents=True,exist_ok=True)
    sys.path[:0]=['scripts','AgenticArxiv']
    from benchmark.tasks_expanded import get_expanded_tasks
    from generate_parametric_sft_data import build_parametric_tasks
    original=get_expanded_tasks();byid={t['id']:t for t in original}
    split=json.loads(Path('data/splits/v3_81.json').read_text())['split']
    train=set(split['train']);held=set(split['iid_test']+split['ood_test'])
    train_specs=[t for t in original if t['id'] in train]+[d.spec.to_task() for d in build_parametric_tasks(include_t5=True)]
    previous=json.loads(Path('artifacts/reproduction/round-007-2026-10-04/fresh-tasks.json').read_text())
    known_signatures={signature(t) for t in original+train_specs+previous}
    changes={
      'search_CV_1d_1':('请列出cs.CV最近13天的论文，最多2篇。',[{'aspect':'CV','days':13,'max_results':2}],None),
      'search_RO_30d_25':('在机器人学cs.RO中找最近23天的论文，返回上限7篇。',[{'aspect':'RO','days':23,'max_results':7}],None),
      'search_CR_14d_10':('查找cs.CR最近17天发表的论文，最多给4篇。',[{'aspect':'CR','days':17,'max_results':4}],None),
      'search_kw_rag':('用all:retrieval augmented generation检索最近30天的arXiv论文，最多3篇。',[{'query':'all:retrieval augmented generation','max_results':3,'days':30}],None),
      'opt_service':('请用google服务翻译会话中的第2篇论文。',[{'ref':2,'service':'google'}],[{'name':'get_recently_submitted_cs_papers','args':{'aspect':'AI','days':13,'max_results':4}},{'name':'download_arxiv_pdf','args':{'ref':2}}]),
      'ref_stress_learning_state':('把标题中包含完整短语“Handover of In-Context Learning State”的那篇论文下载下来。',[{'ref':'Handover of In-Context Learning State'}],[{'name':'get_recently_submitted_cs_papers','args':{'aspect':'AI','days':13,'max_results':5}}]),
      'ref_ctrl_id_cache':('请检查论文2608.14546v1的缓存状态。',[{'ref':'2608.14546v1'}],[{'name':'get_recently_submitted_cs_papers','args':{'aspect':'CV','days':13,'max_results':5}}]),
      'ref_ctrl_null_cache':('查看刚才操作的那篇论文目前缓存状态。',[{'ref':None}],[{'name':'get_recently_submitted_cs_papers','args':{'aspect':'AI','days':13,'max_results':5}},{'name':'download_arxiv_pdf','args':{'ref':4}}]),
      'multi_cl5_dl2_cache':('查cs.CL最近13天的4篇论文，下载第3篇，然后检查第3篇缓存。',[{'aspect':'CL','days':13,'max_results':4},{'ref':3},{'ref':3}],None),
      'multi_ai5_dl1_tr':('找cs.AI最近13天的4篇论文，下载第2篇，再翻译第2篇。',[{'aspect':'AI','days':13,'max_results':4},{'ref':2},{'ref':2}],None),
      'multi_ro3_dl_two':('查cs.RO最近17天的4篇论文，分别下载第2篇和第3篇。',[{'aspect':'RO','days':17,'max_results':4},{'ref':2},{'ref':3}],None),
      'multi_cv5_dl1_cache_tr':('找cs.CV最近13天的4篇论文，下载第2篇，检查第2篇缓存后再翻译第2篇。',[{'aspect':'CV','days':13,'max_results':4},{'ref':2},{'ref':2},{'ref':2}],None),
      'multi_ai3_tr1':('检索cs.AI最近17天的4篇论文，随后直接翻译第3篇。',[{'aspect':'AI','days':17,'max_results':4},{'ref':3}],None),
      'state_ref_cache_active':('查一下刚下载的论文现在是否已有缓存。',[{'ref':None}],[{'name':'get_recently_submitted_cs_papers','args':{'aspect':'AI','days':13,'max_results':5}},{'name':'download_arxiv_pdf','args':{'ref':5}}]),
      'state_cache_two':('分别查询第2篇和第4篇的缓存状态。',[{'ref':2},{'ref':4}],[{'name':'get_recently_submitted_cs_papers','args':{'aspect':'AI','days':13,'max_results':5}}]),
      'qa_cr5_abstract':('检索cs.CR最近17天的4篇论文，下载第3篇，再读取第3篇的abstract摘要章节。',[{'aspect':'CR','days':17,'max_results':4},{'ref':3},{'ref':3,'section':'abstract'}],None),
      'summary_lg5_tldr120':('检索cs.LG最近13天的4篇论文，下载第2篇，以tldr风格在180词以内总结第2篇。',[{'aspect':'LG','days':13,'max_results':4},{'ref':2},{'ref':2,'style':'tldr','max_words':180}],None),
      'figure_ai5_ref4':('查找cs.AI最近13天的5篇论文，下载第5篇，然后提取第5篇图表。',[{'aspect':'AI','days':13,'max_results':5},{'ref':5},{'ref':5}],None),
      'analyze_ai5_ref4_desc':('查询cs.AI最近13天的5篇论文，下载第5篇，提取图表并describe第1张图的内容。',[{'aspect':'AI','days':13,'max_results':5},{'ref':5},{'ref':5},{'ref':5,'figure_no':1,'question':'describe'}],None),
      'constraint_translate_only':('只将刚下载的论文翻译成双语版本；不要重新检索，也不要查询缓存。',[{'ref':None,'keep_dual':True}],[{'name':'get_recently_submitted_cs_papers','args':{'aspect':'AI','days':13,'max_results':4}},{'name':'download_arxiv_pdf','args':{'ref':2}}]),
      'infeasible_index_out_of_range':('请下载当前列表的第8篇论文。',[],[{'name':'get_recently_submitted_cs_papers','args':{'aspect':'AI','days':13,'max_results':4}}]),
      'chain_cl5_dl3_tr3':('找到cs.CL最近13天的4篇论文，下载第2篇，先查第2篇缓存，再翻译第2篇。',[{'aspect':'CL','days':13,'max_results':4},{'ref':2},{'ref':2},{'ref':2}],None),
    }
    assert set(changes)==held
    fresh=[];groups={'iid_test':[],'ood_test':[]};seen=set()
    for parent in sorted(held):
        text,args,setup=changes[parent];t=copy.deepcopy(byid[parent]);t.pop('note',None);t.pop('expected_paper_ids',None)
        t.update(id='fresh008_'+parent,task=text,expected_tool_args=args,parent_task_id=parent)
        if setup is not None:t['setup']=setup
        assert len(args)==len(t['expected_tools'])
        key=signature(t);assert key not in known_signatures,parent;assert key not in seen,parent;seen.add(key)
        group='iid_test' if parent in split['iid_test'] else 'ood_test';groups[group].append(t['id']);fresh.append(t)
    assert len(fresh)==22 and not held&train
    os.environ['STORE_BACKEND']='memory'
    with tempfile.TemporaryDirectory(prefix='round008-gold-') as scratch:
        for key in ['PDF_RAW_PATH','PDF_TRANSLATED_PATH','PDF_TRANSLATED_LOG_PATH','PDF_CACHE_PATH','TRANSLATE_CACHE_PATH','PDF_FIGURES_PATH']:os.environ[key]=str(Path(scratch)/key.lower())
        from benchmark.semantic_oracle import attach_expected_paper_ids,_call
        from benchmark.task_spec import reference_terminal_thought
        from benchmark.metrics import is_strict_success
        from rl.multiturn_env import AgenticArxivMultiTurnEnv
        from rl.reward import RewardCalculator
        from tools.bootstrap import require_all_tools
        require_all_tools('Round008 fresh gold validation')
        reward=RewardCalculator(curriculum_steps=0);oracles=[];results=[];traces=[]
        for original_task in fresh:
            task=attach_expected_paper_ids([original_task],SNAPSHOT)[0]
            env=AgenticArxivMultiTurnEnv(SNAPSHOT);env.reset(task_id=task['id'])
            history=[]
            for setup in task.get('setup',[]):_call(env,setup['name'],setup.get('args',{}))
            for name,args in zip(task['expected_tools'],task['expected_tool_args']):
                obs=_call(env,name,args)
                if isinstance(obs,list):assert not any(x.get('_mock_env',{}).get('offline_fallback') for x in obs if isinstance(x,dict)),task['id']
                history.append({'thought':'执行预先声明的工具步骤','action':json.dumps({'name':name,'args':args},ensure_ascii=False),'observation':json.dumps(obs,ensure_ascii=False,default=str)})
            history.append({'thought':reference_terminal_thought(task),'action':'FINISH','observation':'任务完成'})
            breakdown,metric=reward.compute_reward_breakdown(task,{'history':history})
            results.append({'task_id':task['id'],'strict_success':is_strict_success(metric),'reward':breakdown.total})
            oracles.append(task);traces.append({'task_id':task['id'],'history':history})
        write(R/'fresh-gold-validation.json',{'tasks':22,'passed':sum(x['strict_success'] and x['reward']>=.99 for x in results),'model_inference':False,'snapshot_sha256':sha(SNAPSHOT),'results':results})
        assert all(x['strict_success'] and x['reward']>=.99 for x in results),'Gold resource validation failed; no model evaluation allowed'
    write(R/'fresh-tasks.json',fresh);write(R/'fresh-oracles.json',oracles);write(R/'fresh-splits.json',{'split':groups})
    (R/'fresh-gold-traces.jsonl').write_text(''.join(json.dumps(t,ensure_ascii=False)+'\n' for t in traces))
    write(R/'fresh-data-manifest.json',{'policy':'Predeclared deterministic parameter recombinations of existing held-out parents; no model-output filtering. Same known-paper world, not new-paper generalization.',
      'fresh_tasks':22,'split_counts':{k:len(v) for k,v in groups.items()},'heldout_parent_overlap_with_train':0,'exact_contract_overlap_with_original81_SFT137_and_round007':0,'exact_contract_duplicates':0,'old_heldout_parents':sorted(held),'frozen_before_model_evaluation':True,'selection_from_round008_model_outputs':False,
      'sha256':{str(p):sha(p) for p in [R/'fresh-tasks.json',R/'fresh-oracles.json',R/'fresh-splits.json',SNAPSHOT,Path('data/repro_20261002/sft_train.jsonl')]}})
    print('Frozen fresh tasks and oracle: gold22/22, contracts overlap0',flush=True)

if __name__=='__main__':main()
