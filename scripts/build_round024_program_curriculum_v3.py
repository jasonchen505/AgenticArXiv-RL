#!/usr/bin/env python3
"""Train/schema-only compositional curriculum; CPU resource/gold gate, no learning."""
from pathlib import Path
import collections,copy,datetime,hashlib,itertools,json,os,sys,tempfile
R=Path('artifacts/reproduction/round-024-2026-10-07');D=Path('data/repro_round024');OLD=R.parent/'round-022-2026-10-07'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def state(phase,**kw):write(R/'curriculum-status.json',dict(phase=phase,updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),GPU_jobs=0,new_training_steps=0,**kw))
N={'C':'get_paper_content','S':'summarize_paper','F':'extract_paper_figures','A':'analyze_figure'}
FAMILIES={'train':['CF','SF','FC','FS','CFA','SFA','FAC','FAS'],'dev':['CSF','SCF'],'confirm':['CSFA','SCFA']}
def signature(f):return ['get_recently_submitted_cs_papers','download_arxiv_pdf']+[N[x] for x in f]
def main():
 R.mkdir(parents=True,exist_ok=True);D.mkdir(parents=True,exist_ok=True);assert not (R/'curriculum-preparation-plan.json').exists()
 source=Path('data/repro_round014/certified_combined.jsonl');rows=[json.loads(x) for x in source.read_text().splitlines()];experts=rows[:398];assert len(rows)==443
 trainpath=R.parent/'round-013-2026-10-05/collection-oracles.json';train=read(trainpath);knownpath=R.parent/'round-002-2026-10-02/task-oracles.json';known=read(knownpath)
 fullprogs={tuple(t['expected_tools']) for t in train+known};assert len(train)==137
 for split,fs in FAMILIES.items():
  for f in fs:assert tuple(signature(f)) not in fullprogs,(split,f)
 allf=sum(FAMILIES.values(),[]);assert len(allf)==len(set(allf))==12
 # All values/tool names derive from TRAIN tasks; heldout file is used solely to exclude full-program overlap.
 domains=collections.defaultdict(set)
 for t in train:
  for name,a in zip(t['expected_tools'],t['expected_tool_args']):
   for k,v in a.items():domains[(name,k)].add(json.dumps(v,sort_keys=True))
 required={('get_recently_submitted_cs_papers','aspect'):['AI','CV'],('get_recently_submitted_cs_papers','days'):[7],('get_recently_submitted_cs_papers','max_results'):[5],('download_arxiv_pdf','ref'):[1,3],('get_paper_content','section'):[None],('summarize_paper','style'):['tldr','structured'],('summarize_paper','max_words'):[60,120],('analyze_figure','figure_no'):[1],('analyze_figure','question'):['describe']}
 for k,values in required.items():assert all(json.dumps(v,sort_keys=True) in domains[k] for v in values),(k,values)
 sources=[source,trainpath,knownpath,Path(__file__),Path('scripts/generate_sft_data.py'),Path('AgenticArxiv/rl/env.py'),Path('AgenticArxiv/agents/prompt_templates.py'),Path('AgenticArxiv/benchmark/metrics.py'),Path('AgenticArxiv/benchmark/semantic_oracle.py')]
 assets={}
 for world in ['trainA','trainB']:
  sources.append(Path('data/repro_round022_v2')/(world+'-snapshot.json'))
  for mf in [R.parent/'round-021-2026-10-06'/(world+'-asset-manifest.json'),OLD/(world+'-reserve-manifest.json')]:
   sources.append(mf)
   for p,h in read(mf)['sha256'].items():assert sha(p)==h,p;assets[p]=h
 plan={'registered_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_only':True,'hypothesis':'Paper-world inputs expanded without new ordered tool programs. Test train/schema-derived compositions, not another paper/year/loss sweep. This is a candidate intervention, not proof of the bottleneck.','families':FAMILIES,'parameters':{'aspects':['AI','CV'],'refs':[1,3],'summary_budgets':[60,120],'days':7,'max_results':5,'section':None,'figure_no':1,'question':'describe'},'splits':{'train':{'world':'trainA','tasks':64,'families':8},'dev':{'world':'trainA','tasks':16,'families':2},'confirmA':{'world':'trainA','tasks':16,'families':2},'confirmB':{'world':'trainB','tasks':16,'families':2}},'teacher_format':'Same original expert description; list count-only, str(result)[:500], generated deterministic expert Thoughts. Genuine unchanged snapshot tool outputs, full sequences end FINISH.','source_constraint':'Only TRAIN tool and argument domains generate candidates. Prior task tool-sequences are exclusion-only; no heldout text/parameters/trajectories/model failures consumed.','program_isolation':'Full ordered tool signatures disjoint across train/dev/confirm and from all original137train+81benchmark programs. ConfirmA/B share 2families across two old training paperworlds; parameter/tool/subsequence overlap intentional. Not independent16programs, not newpaper-blind worlds.','resource_admission':'Before any new gold/model outputs, copy both old train snapshots; for each AI/CV pool use first original-index lexicographic membership-preserving permutation with default content and >=1 real figure at slots1/3. This derives from the uniform train/schema curriculum, never model/heldout failures. Preserve failedv2 resources/gold/source/plan. Original assets unchanged, no replacement or fabricated figures.', 'gate':'All112 CPU gold must pass without replacements; every actual PDF/image and frozen source SHA must stay unchanged, all train rows reconstructed, tokenizer<4096. Abort/preserve failures. No model inference or learning until independent audit and complete matched-budget numerical learning protocol.','GPU_jobs':0,'new_training_steps':0,'learning_permitted':False,'sha256':{str(p):sha(p) for p in sources},'asset_sha256':assets}
 write(R/'curriculum-preparation-plan.json',plan)
 os.environ.update(CUDA_VISIBLE_DEVICES='',STORE_BACKEND='memory',FIGURE_ANALYSIS_BACKEND='extractive')
 with tempfile.TemporaryDirectory(prefix='curriculum024-') as tmp:
  for k,n in [('PDF_RAW_PATH','raw'),('PDF_TRANSLATED_PATH','translated'),('PDF_FIGURES_PATH','figures')]:os.environ[k]=str(Path(tmp)/n)
  sys.path[:0]=['AgenticArxiv','scripts']
  from tools.bootstrap import require_all_tools
  require_all_tools('Round024 train-only program curriculum')
  from models.store import use_memory_store
  from rl.env import MockArxivEnv
  from agents.side_effects import LocalSideEffectManager
  from agents.prompt_templates import build_visible_setup_context
  from generate_sft_data import execute_expert_tool,build_sft_samples_from_history
  from benchmark.semantic_oracle import attach_expected_paper_ids
  from benchmark.metrics import extract_metrics,is_strict_success
  from transformers import AutoTokenizer
  resource=[]
  for world in ['trainA','trainB']:
   original=Path('data/repro_round022_v2')/(world+'-snapshot.json');snapshot=read(original)
   content={e['result']['paper_id'] for e in snapshot['get_paper_content'].values() if e['result'].get('section') is None and e['result'].get('content','').strip()}
   figures={e['result']['paper_id']:e['result']['count'] for e in snapshot['extract_paper_figures'].values()}
   for e in snapshot['get_recently_submitted_cs_papers'].values():
    if e['args']['aspect'] not in ['AI','CV']:continue
    papers=e['result']
    def assign(chosen):
     if len(chosen)==len(papers):return chosen
     slot=len(chosen)+1
     for i,paper in enumerate(papers):
      if i in chosen:continue
      if slot in [1,3] and not (paper['id'] in content and figures.get(paper['id'],0)>=1):continue
      ans=assign(chosen+[i])
      if ans is not None:return ans
    perm=assign([]);assert perm is not None,(world,e['args']['aspect'])
    before=[p['id'] for p in papers];e['result']=[papers[i] for i in perm]
    resource.append({'world':world,'aspect':e['args']['aspect'],'before':before,'after':[p['id'] for p in e['result']],'permutation_one_based':[i+1 for i in perm],'membership_preserved':True})
   write(D/(world+'-snapshot.json'),snapshot)
  write(R/'curriculum-resource-admission.json',{'rule':plan['resource_admission'],'permutations':resource,'model_outputs':0,'original_assets_unchanged':True,'sha256':{str(D/(w+'-snapshot.json')):sha(D/(w+'-snapshot.json')) for w in ['trainA','trainB']}})
  desc=experts[0]['prompt'][0]['content'].split('你有以下工具可以使用：\n\n',1)[1].split('\n\n当前任务：',1)[0];generated=[];results={};alltasks=[]
  for split,sc in plan['splits'].items():
   tasks=[];families=FAMILIES['confirm' if split.startswith('confirm') else split]
   for f,(aspect,ref,words) in itertools.product(families,itertools.product(['AI','CV'],[1,3],[60,120])):
    style='tldr' if words==60 else 'structured';identifier=f'curr024_{split}_{f}_{aspect}_{ref}_{words}';arguments=[{'aspect':aspect,'days':7,'max_results':5},{'ref':ref}]
    steptext={'C':'读取它默认返回的标题与摘要','S':f'以{style}风格用不超过{words}词总结它','F':'抽取它的全部图表','A':'描述它的第1张图'}
    for letter in f:arguments.append({'C':{'ref':ref,'section':None},'S':{'ref':ref,'style':style,'max_words':words},'F':{'ref':ref},'A':{'ref':ref,'figure_no':1,'question':'describe'}}[letter])
    tasks.append({'id':identifier,'parent_task_id':'curr024_family_'+f,'task':f'检索最近7天'+('人工智能(cs.AI)' if aspect=='AI' else '计算机视觉(cs.CV)')+f'论文5篇，下载第{ref}篇，然后按以下顺序处理同一篇论文：'+'；'.join(steptext[x] for x in f)+'。完成这些步骤后结束。','expected_tools':signature(f),'expected_tool_args':arguments,'expected_termination':'FINISH','category':'program_curriculum','difficulty':'hard','requires_offline':True,'max_iterations':len(signature(f))+2,'provenance':{'generator':'train/schema tool grammar, no heldout traces','family':f,'parameter_domain_from_train':True,'world':sc['world']}})
   assert len(tasks)==sc['tasks'];snapshot=D/(sc['world']+'-snapshot.json');oracles=attach_expected_paper_ids(tasks,snapshot);write(R/(split+'-tasks.json'),tasks);write(R/(split+'-oracles.json'),oracles);alltasks+=tasks
   gold=[]
   for t in oracles:
    state('CPU_gold',split=split,current_task=t['id'],completed_in_split=len(gold));use_memory_store(reset=True);env=MockArxivEnv(snapshot,mode='replay',offline_download=True);side=LocalSideEffectManager();session='gold_'+t['id'];history=[]
    for i,(name,args) in enumerate(zip(t['expected_tools'],t['expected_tool_args'])):
     result=execute_expert_tool(env,side,name,args,session)
     if isinstance(result,dict):assert result.get('paper_id')==t['expected_paper_ids'][i],(t['id'],name)
     if name=='extract_paper_figures':assert result['count']>=1
     if name=='get_paper_content':assert result['content'].strip()
     obs=f'成功获取 {len(result)} 篇论文' if isinstance(result,list) else str(result)[:500]
     history.append({'thought':f'需要调用 {name} 来处理任务步骤 {i+1}','action':json.dumps({'name':name,'args':args},ensure_ascii=False),'observation':obs})
    history.append({'thought':'所有步骤均已成功执行，任务已完成','action':'FINISH','observation':'任务完成'})
    m=extract_metrics(t,{'history':history},'CPU_curriculum_gold',0,session);assert is_strict_success(m),(t['id'],m.to_dict());gold.append({'task_id':t['id'],'history':history,'metrics':m.to_dict()})
    if split=='train':
     samples=build_sft_samples_from_history(t['task'],desc,history,source_task_id=t['id'],source_split='train',initial_history=build_visible_setup_context(t))
     for i,g in enumerate(samples):generated.append({'prompt':[g['messages'][0]],'chosen':[g['messages'][1]],'rejected':[copy.deepcopy(g['messages'][1])],'source_task_id':t['id'],'parent_task_id':t['parent_task_id'],'trajectory_step':i,'pair_type':'expert_replay','program_family':t['provenance']['family'],'training_world':'trainA','provenance':t['provenance']})
   (R/(split+'-gold-traces.jsonl')).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in gold));results[split]={'tasks':len(gold),'gold_passed':len(gold),'program_families':len(families)}
  assert len(generated)==352;tok=AutoTokenizer.from_pretrained('outputs/repro_20261003/sft_e1/final',local_files_only=True);lengths=[]
  for x in generated:
   ids=tok.apply_chat_template(x['prompt']+x['chosen'],tokenize=True);ids=ids['input_ids'] if hasattr(ids,'keys') else ids;lengths.append(len(ids))
  assert max(lengths)<4096,max(lengths)
  p=D/'curriculum-experts.jsonl';p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in generated));write(R/'all-tasks.json',alltasks)
  for p,h in dict(plan['sha256'],**plan['asset_sha256']).items():assert sha(p)==h,p
  write(R/'curriculum-gold-validation.json',{'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'splits':results,'gold_total':112,'gold_passed':112,'new_train_expert_decisions':352,'new_train_program_signatures':8,'new_dev_program_signatures':2,'new_confirmation_program_signatures':2,'full_program_overlap_train_dev_confirm':0,'original_benchmark_and_train_program_overlap':0,'all_sources_and_assets_unchanged':True,'token_lengths':{'maximum':max(lengths),'mean':sum(lengths)/len(lengths),'sequences':len(lengths),'truncation':0},'GPU_jobs':0,'new_training_steps':0,'learning_permitted':False,'curriculum_data_sha256':sha(D/'curriculum-experts.jsonl'),'complete_matched_learning_protocol_required':True})
  state('CPU_candidate_ready_not_learning',gold_passed=112,train_rows=352,learning_permitted=False)
if __name__=='__main__':
 try:main()
 except Exception as e:state('failed',error=str(e));raise
