#!/usr/bin/env python3
from pathlib import Path
import os,sys,json,hashlib,datetime,itertools,tempfile
R=Path('artifacts/reproduction/round-029-2026-10-09');OLD=R.parent/'round-024-2026-10-07';read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
N={'C':'get_paper_content','S':'summarize_paper','F':'extract_paper_figures','A':'analyze_figure'}
def signature(f):return ['get_recently_submitted_cs_papers','download_arxiv_pdf']+[N[c] for c in f]
def main():
 os.environ.update(CUDA_VISIBLE_DEVICES='',STORE_BACKEND='memory',FIGURE_ANALYSIS_BACKEND='extractive');R.mkdir(exist_ok=True);assert not (R/'fresh-preparation-plan.json').exists();audit=read(R/'probe-independent-audit.json');assert audit['passed'] and audit['eligibility_gate_met']
 sourcefiles=[R.parent/'round-002-2026-10-02/task-oracles.json',R.parent/'round-013-2026-10-05/collection-oracles.json',OLD/'all-tasks.json']+[R.parent/'round-026-2026-10-08'/(g+'-oracles.json') for g in ['confirmA','confirmB']];previous=[t for p in sourcefiles for t in read(p)];excluded={tuple(t['expected_tools']) for t in previous};candidates=sorted(''.join(x) for x in itertools.permutations('CSFA') if x.index('F')<x.index('A'));families=[f for f in candidates if tuple(signature(f)) not in excluded][:2];assert len(families)==2
 prep={'registered_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_only':True,'families':families,'rule':'EachC,S,F,A once, FbeforeA;lexicographically firsttwo absent originaltrain/benchmark/R024all/R026confirm full sequences. Exclusion only; noheldoutfailuretext/params/rollouts generate training or newconfirmation.','selection_before_gold_and_model_predictions':True,'parameters_source':'OriginalTRAIN domains AI/CV,ref1/3,summary60/120,days7,results5,sectionNone,figure1,describe','worlds':['trainA','trainB'],'tasks_per_world':16,'gold_planned':32,'scope':'Two new full programs, sharedtools/params/subsequences and reusedTRAINpaperworlds. Notpaperblind norbroadindependentprogramgeneralization. R024andR026confirmnowdiagonly.','sha256':{str(p):sha(p) for p in sourcefiles+[R/'probe-independent-audit.json',Path(__file__)]}}
 write(R/'fresh-preparation-plan.json',prep);sys.path[:0]=['AgenticArxiv','scripts'];from tools.bootstrap import require_all_tools;from benchmark.semantic_oracle import attach_expected_paper_ids;from rl.multiturn_env import AgenticArxivMultiTurnEnv;from benchmark.semantic_oracle import _call;from benchmark.task_spec import reference_terminal_thought;from benchmark.metrics import extract_metrics,is_strict_success
 require_all_tools('Round029 prospective online confirmation');gold=[]
 with tempfile.TemporaryDirectory(prefix='r029-gold-') as tmp:
  for k,n in [('PDF_RAW_PATH','raw'),('PDF_TRANSLATED_PATH','translated'),('PDF_FIGURES_PATH','figures')]:os.environ[k]=str(Path(tmp)/n)
  for group,world in [('freshA','trainA'),('freshB','trainB')]:
   tasks=[]
   for f,(aspect,ref,words) in itertools.product(families,itertools.product(['AI','CV'],[1,3],[60,120])):
    style='tldr' if words==60 else 'structured';spec={'C':{'ref':ref,'section':None},'S':{'ref':ref,'style':style,'max_words':words},'F':{'ref':ref},'A':{'ref':ref,'figure_no':1,'question':'describe'}};phrases={'C':'读取它默认返回的标题与摘要','S':f'以{style}风格用不超过{words}词总结它','F':'抽取它的全部图表','A':'描述它的第1张图'}
    tasks.append({'id':f'online029_{group}_{f}_{aspect}_{ref}_{words}','parent_task_id':'online029_family_'+f,'task':'检索最近7天'+('人工智能(cs.AI)' if aspect=='AI' else '计算机视觉(cs.CV)')+f'论文5篇，下载第{ref}篇，然后按以下顺序处理同一篇论文：'+'；'.join(phrases[c] for c in f)+'。完成这些步骤后结束。','expected_tools':signature(f),'expected_tool_args':[{'aspect':aspect,'days':7,'max_results':5},{'ref':ref}]+[spec[c] for c in f],'expected_termination':'FINISH','category':'online_program_confirmation','difficulty':'hard','requires_offline':True,'max_iterations':8,'provenance':{'family':f,'world':world,'source':'TRAIN/schema grammar-selected, prospective confirmation only'}})
   snapshot=Path('data/repro_round024')/(world+'-snapshot.json');oracles=attach_expected_paper_ids(tasks,snapshot);assert len({json.dumps([t['task'],t['expected_tools'],t['expected_tool_args']],ensure_ascii=False,sort_keys=True) for t in tasks})==16;write(R/(group+'-tasks.json'),tasks);write(R/(group+'-oracles.json'),oracles)
   for t in oracles:
    env=AgenticArxivMultiTurnEnv(snapshot);env.reset(task_id=t['id']);history=[]
    for n,a in zip(t['expected_tools'],t['expected_tool_args']):
     z=_call(env,n,a);history.append({'thought':'按前瞻任务契约执行','action':json.dumps({'name':n,'args':a},ensure_ascii=False),'observation':json.dumps(z,ensure_ascii=False,default=str)})
    history.append({'thought':reference_terminal_thought(t),'action':'FINISH','observation':'任务完成'});m=extract_metrics(t,{'history':history},'CPU_R029_gold',0,t['id']);assert is_strict_success(m) and not m.false_finish;gold.append({'task_id':t['id'],'group':group,'history':history,'metrics':m.to_dict()})
 assert len(gold)==32;(R/'fresh-gold-traces.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in gold));write(R/'fresh-split.json',{'split':{g:[t['id'] for t in read(R/(g+'-tasks.json'))] for g in ['freshA','freshB']}})
 write(R/'fresh-preparation-result.json',{'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'CPU_only':True,'families':families,'gold_passed':32,'model_predictions':0,'new_learning_steps':0,'full_program_overlap_previous':0,'unique_visible_contracts_per_world':16,'sha256':{str(p):sha(p) for p in R.glob('fresh-*') if p.is_file()}});print(json.dumps({'families':families,'gold_passed':32,'model_predictions':0}))
if __name__=='__main__':main()
