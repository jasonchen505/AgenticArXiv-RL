#!/usr/bin/env python3
"""CPU-only preparation: original TRAIN anchors, terminal exposure audit, fresh program gold."""
from pathlib import Path
import collections,datetime,hashlib,itertools,json,os,sys,tempfile
R=Path('artifacts/reproduction/round-026-2026-10-08');OLD=R.parent/'round-024-2026-10-07';D=Path('data/repro_round026')
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def action(x):
 a=x['chosen'][0]['content'].split('Action: ',1)[1].strip();return 'FINISH' if a=='FINISH' else json.loads(a)['name']
N={'C':'get_paper_content','S':'summarize_paper','F':'extract_paper_figures','A':'analyze_figure'}
def signature(f):return ['get_recently_submitted_cs_papers','download_arxiv_pdf']+[N[x] for x in f]
def main():
 R.mkdir(exist_ok=True);D.mkdir(exist_ok=True);assert not (R/'retention-preparation-plan.json').exists();closure=read(R/'closure-independent-audit.json');assert closure['passed'] and closure['fixed_recipe_closed']
 trainpath=R.parent/'round-013-2026-10-05/collection-oracles.json';train={t['id']:t for t in read(trainpath)};source=Path('data/repro_round014/certified_combined.jsonl');original=rows(source);assert len(original)==443 and all(x['source_task_id'] in train for x in original)
 oldtasks=read(R.parent/'round-002-2026-10-02/task-oracles.json')+list(train.values())+read(OLD/'all-tasks.json');excluded={tuple(t['expected_tools']) for t in oldtasks}
 # Grammar rule before gold/model evaluation: each operation once; analysis after extraction;
 # lexicographically first two valid unseen full sequences. No failure-targeted family choice.
 candidates=sorted(''.join(x) for x in itertools.permutations('CSFA') if x.index('F')<x.index('A'));families=[f for f in candidates if tuple(signature(f)) not in excluded][:2];assert len(families)==2
 frozen=read(OLD/'experiment-plan.json')['sha256']
 for p,h in frozen.items():assert sha(p)==h,p
 prep={'registered_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_only':True,'new_training_steps':0,'learning_permitted':False,'purpose':'Prepare a retention-constrained curriculum study after positive limited-composition transfer and failed old-task retention. No further data/year/reward sweep.','anchor_rule':'All443 original train-only chosen decisions,398experts+45certifiederrors, one frozen source-SFT teacher; no old/new Dev or confirmation traces become training data.','future_method':'Chosen CE on unchanged795row curriculum/repeat plus completion-token forward KL(sourceSFT||policy) on a shared originalTRAIN anchor stream. Fixed coefficient proposed0.1, no coefficient sweep. Four arms/two seeds. This is a design proposal, NOT a registered executable learning protocol.','fresh_confirmation_selection':'Each ofC,S,F,A once;F precedesA; lexicographically first two full sequences absent all original train/benchmark and R024 splits. Existing R024confirmations now diagnostic-only, not fresh confirmation.','families':families,'splits':{'confirmA':{'world':'trainA','tasks':16},'confirmB':{'world':'trainB','tasks':16}},'parameters_from_original_train':{'aspects':['AI','CV'],'refs':[1,3],'summary_budgets':[60,120],'days':7,'max_results':5,'section':None,'figure_no':1,'question':'describe'},'scope':'Only two new full ordered programs, shared tools/arguments/subsequences, two reused training paperworlds; no paper-blind or broad independent-family generalization. New tasks are a prospective confirmation candidate, never generate teacher anchors.','launch_gates':['Independent gold/anchor/source audit','Trainer source completion mask and KL normalization verified;equal policy/teacher KL0; teacher frozen and no gradient','Memory preflight with zero optimizer steps; onlyGPU4/7 and23000MiBfree+UUIDcompute0','Full exact budget, seeds, optimizer, final checkpoint, numeric ALL success rule and all model/data/resource/source hashes registered before any learning','All final checkpoints frozen before any candidate/fresh confirmation inference'],'scientific_inputs_sha256':frozen,'source_sha256':{str(source):sha(source),str(trainpath):sha(trainpath),str(Path(__file__)):sha(Path(__file__))}}
 write(R/'retention-preparation-plan.json',prep)
 anchors=[]
 for i,x in enumerate(original):anchors.append({'anchor_index':i,'source_task_id':x['source_task_id'],'pair_type':x['pair_type'],'terminal_mode':train[x['source_task_id']].get('expected_terminal_mode','completed'),'prompt':x['prompt'],'chosen':x['chosen']})
 (D/'source-train-anchors.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in anchors))
 terminal={}
 for name in ['curriculum','matched_repeat']:
  data=rows(Path('data/repro_round024')/(name+'_combined.jsonl'));added=data[398:750];modes=collections.Counter()
  for x in added:
   if action(x)=='FINISH':modes[train.get(x['source_task_id'],{}).get('expected_terminal_mode','completed')]+=1
  terminal[name]={'added_rows':len(added),'added_action_marginals':dict(collections.Counter(map(action,added))),'added_FINISH_semantic_modes':dict(modes),'original398_prefix_identical':data[:398]==original[:398],'original45_tail_identical':data[-45:]==original[-45:]}
 assert terminal['curriculum']['added_action_marginals']==terminal['matched_repeat']['added_action_marginals']
 write(R/'train-terminal-exposure-audit.json',{'CPU_only':True,'training_only':True,'data_unchanged':True,'counts':terminal,'interpretation':'Matched action-tool marginals do not imply matched terminal semantic modes: curriculum64 completed vs replay62completed+2blocked. This is a nuisance component of the intervention, not proof it caused retention loss. Future KL comparison retains same frozen data so it tests retention within each data arm;course-vs-repeat still not a pure tool-order causal claim.'})
 os.environ.update(CUDA_VISIBLE_DEVICES='',STORE_BACKEND='memory',FIGURE_ANALYSIS_BACKEND='extractive')
 gold=[];taskfiles={}
 with tempfile.TemporaryDirectory(prefix='retention026-') as tmp:
  for k,n in [('PDF_RAW_PATH','raw'),('PDF_TRANSLATED_PATH','translated'),('PDF_FIGURES_PATH','figures')]:os.environ[k]=str(Path(tmp)/n)
  sys.path[:0]=['AgenticArxiv','scripts']
  from tools.bootstrap import require_all_tools
  require_all_tools('Round026 CPU prospective confirmation')
  from models.store import use_memory_store
  from rl.env import MockArxivEnv
  from agents.side_effects import LocalSideEffectManager
  from generate_sft_data import execute_expert_tool
  from benchmark.semantic_oracle import attach_expected_paper_ids
  from benchmark.metrics import extract_metrics,is_strict_success
  from transformers import AutoTokenizer
  for split,sc in prep['splits'].items():
   tasks=[]
   for f,(aspect,ref,words) in itertools.product(families,itertools.product(['AI','CV'],[1,3],[60,120])):
    style='tldr' if words==60 else 'structured';args=[{'aspect':aspect,'days':7,'max_results':5},{'ref':ref}]+[{'C':{'ref':ref,'section':None},'S':{'ref':ref,'style':style,'max_words':words},'F':{'ref':ref},'A':{'ref':ref,'figure_no':1,'question':'describe'}}[c] for c in f]
    words_map={'C':'读取它默认返回的标题与摘要','S':f'以{style}风格用不超过{words}词总结它','F':'抽取它的全部图表','A':'描述它的第1张图'}
    tasks.append({'id':f'retain026_{split}_{f}_{aspect}_{ref}_{words}','parent_task_id':'retain026_family_'+f,'task':'检索最近7天'+('人工智能(cs.AI)' if aspect=='AI' else '计算机视觉(cs.CV)')+f'论文5篇，下载第{ref}篇，然后按以下顺序处理同一篇论文：'+'；'.join(words_map[c] for c in f)+'。完成这些步骤后结束。','expected_tools':signature(f),'expected_tool_args':args,'expected_termination':'FINISH','category':'retention_program_confirmation','difficulty':'hard','requires_offline':True,'max_iterations':len(signature(f))+2,'provenance':{'family':f,'world':sc['world'],'source':'original TRAIN domains, grammar-selected, prospective confirmation only'}})
   snapshot=Path('data/repro_round024')/(sc['world']+'-snapshot.json');oracles=attach_expected_paper_ids(tasks,snapshot);write(R/(split+'-tasks.json'),tasks);write(R/(split+'-oracles.json'),oracles);taskfiles[split]=str(R/(split+'-oracles.json'))
   for t in oracles:
    use_memory_store(reset=True);env=MockArxivEnv(snapshot,mode='replay',offline_download=True);side=LocalSideEffectManager();history=[];session='gold_'+t['id']
    for i,(name,a) in enumerate(zip(t['expected_tools'],t['expected_tool_args'])):
     result=execute_expert_tool(env,side,name,a,session)
     if isinstance(result,dict):assert result.get('paper_id')==t['expected_paper_ids'][i]
     if name=='extract_paper_figures':assert result['count']>=1
     if name=='get_paper_content':assert result['content'].strip()
     history.append({'thought':f'需要调用 {name} 来处理任务步骤 {i+1}','action':json.dumps({'name':name,'args':a},ensure_ascii=False),'observation':f'成功获取 {len(result)} 篇论文' if isinstance(result,list) else str(result)[:500]})
    history.append({'thought':'所有步骤均已成功执行，任务已完成','action':'FINISH','observation':'任务完成'});m=extract_metrics(t,{'history':history},'CPU_R026_gold',0,session);assert is_strict_success(m) and not m.false_finish
    gold.append({'split':split,'task_id':t['id'],'history':history,'metrics':m.to_dict()})
  tok=AutoTokenizer.from_pretrained('outputs/repro_20261003/sft_e1/final',local_files_only=True);lengths=[len(tok.apply_chat_template(x['prompt']+x['chosen'],tokenize=True)) for x in anchors];assert max(lengths)<4096
 assert len(gold)==32
 (R/'confirmation-gold-traces.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in gold))
 for p,h in frozen.items():assert sha(p)==h,p
 write(R/'retention-preparation-result.json',{'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_only':True,'new_training_steps':0,'model_predictions':0,'learning_permitted':False,'status':'CPU_prepared_requires_independent_audit_and_executable_protocol','anchor_rows':443,'anchor_experts':398,'anchor_certified_errors':45,'anchor_train_semantics':len(set(x['source_task_id'] for x in anchors)),'anchor_terminal_modes':dict(collections.Counter(x['terminal_mode'] for x in anchors)),'anchor_tokens_max':max(lengths),'truncation':0,'gold_passed':32,'gold_total':32,'full_program_families':families,'full_program_overlap_all_previous':0,'source_and1215_inputs_unchanged':True,'files_sha256':{str(p):sha(p) for p in [D/'source-train-anchors.jsonl',R/'retention-preparation-plan.json',R/'train-terminal-exposure-audit.json',R/'confirmation-gold-traces.jsonl']+[Path(p) for p in taskfiles.values()]}})
 print(json.dumps({'anchors':443,'gold_passed':32,'families':families,'GPU_jobs':0,'learning_permitted':False}))
if __name__=='__main__':main()
