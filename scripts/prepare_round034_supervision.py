import os,sys,json,datetime,collections,itertools
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',STORE_BACKEND='memory',FIGURE_ANALYSIS_BACKEND='extractive');sys.path[:0]=['AgenticArxiv','scripts']
from round034_core import R,D,read,rows,sha,write,encode
from round033_plan_agent import PlanAgent,parse_plan
from tools.bootstrap import require_all_tools
from transformers import AutoTokenizer
from benchmark.runner import BenchmarkRunner
from benchmark.semantic_oracle import attach_expected_paper_ids
from benchmark.metrics import extract_metrics,is_strict_success
from agents.prompt_templates import build_visible_setup_context
from prepare_round033_plan import Fake
N={'C':'get_paper_content','S':'summarize_paper','F':'extract_paper_figures','A':'analyze_figure'}
def signature(f):return ['get_recently_submitted_cs_papers','download_arxiv_pdf']+[N[c] for c in f]
def main():
 assert read(R/'closure-independent-audit.json')['passed'];assert not (R/'supervision-CPU-audit.json').exists();D.mkdir(exist_ok=True);require_all_tools('R034 TRAIN-only supervision and prospective program preparation')
 for k,n in [('PDF_RAW_PATH','raw'),('PDF_TRANSLATED_PATH','translated'),('PDF_FIGURES_PATH','figures')]:os.environ[k]=str((D/'CPU'/n).resolve())
 source='outputs/repro_20261003/sft_e1/final';tok=AutoTokenizer.from_pretrained(source,local_files_only=True);tasks=read(R.parent/'round-033-2026-10-10/TRAIN-unique-tasks.json');train_ids={t['id'] for t in read(R.parent/'round-013-2026-10-05/collection-oracles.json')};anchors=rows('data/repro_round026/source-train-anchors.jsonl');assert len(anchors)==443 and all(x['source_task_id'] in train_ids for x in anchors)
 planrows=[];lengths=[];completion_lengths=[];a=PlanAgent(None)
 for t in tasks:
  assert not t.get('setup') and not t.get('depends_on');prompt=a.plan_request(t['task'],source,build_visible_setup_context(t))['messages'];target={'steps':[{'name':n,'args':v} for n,v in zip(t['expected_tools'],t['expected_tool_args'])],'final_thought':'请求中的工具步骤已完成'};text=json.dumps(target,ensure_ascii=False,separators=(',',':'));assert parse_plan(text)==target
  assert not any(x in json.dumps(prompt,ensure_ascii=False) for x in ['expected_tools','expected_tool_args','provenance',t['id'],'data/repro_round034']);row={'kind':'plan','source_task_id':t['id'],'prompt':prompt,'chosen':[{'role':'assistant','content':text}]};z=encode(tok,row);prefix=tok.apply_chat_template(a.plan_request(t['task'],source,build_visible_setup_context(t))['messages'],add_generation_prompt=True,tokenize=True);assert z['input_ids'][:len(prefix)]==prefix and z['labels'][:len(prefix)]==[-100]*len(prefix);planrows.append(row)
 # Native labels are the original TRAIN-only certified decisions, unchanged.
 native=[dict(kind='native',source_task_id=x['source_task_id'],prompt=x['prompt'],chosen=x['chosen']) for x in anchors];mix=[dict(x,exposure=i) for i in range(8) for x in planrows]+[dict(x,exposure=0) for x in native];assert len(mix)==827
 for x in mix:
  z=encode(tok,x);lengths.append(len(z['input_ids']));completion_lengths.append(sum(t!=-100 for t in z['labels']))
 for name,rs in [('plan-supervision.jsonl',planrows),('mixed-supervision.jsonl',mix)]:assert not (D/name).exists();(D/name).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rs))
 # Select new full programs solely by exclusion, before their gold execution or model prediction.
 files=[R.parent/'round-002-2026-10-02/task-oracles.json',R.parent/'round-013-2026-10-05/collection-oracles.json',R.parent/'round-024-2026-10-07/all-tasks.json']+[R.parent/f'round-{n:03d}-2026-10-{day}'/(g+'-oracles.json') for n,day,gs in [(26,'08',['confirmA','confirmB']),(29,'09',['freshA','freshB']),(31,'09',['freshA','freshB'])] for g in gs]
 excluded={tuple(t['expected_tools']) for f in files for t in read(f)};possible=sorted(''.join(x) for x in itertools.permutations('CSFA') if x.index('F')<x.index('A'));families=[f for f in possible if tuple(signature(f)) not in excluded][:2];assert families==['FSAC','FSCA']
 write(R/'confirmation-preparation-plan.json',{'registered_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'families':families,'selection_before_gold_or_model':True,'selection':'lexicographic first two absent every prior full ordered program, F before A','source_files_sha256':{str(f):sha(f) for f in files},'scope':'shared TRAIN tools/params/subsequences and two old TRAIN paper worlds; not paper-blind or broad generalization','new_learning_steps':0})
 fresh=[]
 for f,(aspect,ref,words) in itertools.product(families,itertools.product(['AI','CV'],[1,3],[60,120])):
  style='tldr' if words==60 else 'structured';spec={'C':{'ref':ref,'section':None},'S':{'ref':ref,'style':style,'max_words':words},'F':{'ref':ref},'A':{'ref':ref,'figure_no':1,'question':'describe'}};phrases={'C':'读取它默认返回的标题与摘要','S':f'以{style}风格用不超过{words}词总结它','F':'抽取它的全部图表','A':'描述它的第1张图'}
  fresh.append({'id':f'plan034_{f}_{aspect}_{ref}_{words}','task':'检索最近7天'+('人工智能(cs.AI)' if aspect=='AI' else '计算机视觉(cs.CV)')+f'论文5篇，下载第{ref}篇，然后按以下顺序处理同一篇论文：'+'；'.join(phrases[c] for c in f)+'。完成这些步骤后结束。','expected_tools':signature(f),'expected_tool_args':[{'aspect':aspect,'days':7,'max_results':5},{'ref':ref}]+[spec[c] for c in f],'expected_termination':'FINISH','category':'full_plan_confirmation','max_iterations':8,'requires_offline':True,'provenance':{'family':f,'source':'TRAIN/schema grammar-selected prospective confirmation only'}})
 assert len(fresh)==16 and not any(tuple(t['expected_tools']) in excluded for t in fresh);write(R/'confirmation-tasks.json',fresh);gold=[]
 for world in ['trainA','trainB']:
  snapshot='data/repro_round024/'+world+'-snapshot.json';runner=BenchmarkRunner(agent_types=['regex'],offline=True,snapshot=snapshot)
  for t in attach_expected_paper_ids(fresh,snapshot):
   runner._reset_offline_trial_state();target={'steps':[{'name':n,'args':v} for n,v in zip(t['expected_tools'],t['expected_tool_args'])],'final_thought':'请求中的工具步骤已完成'};agent=PlanAgent(Fake(json.dumps(target)),side_effect_mgr=runner._side_effects(),env=runner._tool_env());pred,ms=agent.generate(t['task']);z=agent.execute_prediction(t['task'],pred,session_id='CPU_'+t['id']);m=extract_metrics(t,z,'CPU',0,'CPU_'+t['id']);assert is_strict_success(m);gold.append({'world':world,'task_id':t['id'],'history':z['history'],'metrics':m.to_dict()})
 (R/'confirmation-gold.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in gold));write(R/'supervision-CPU-audit.json',{'passed':True,'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'plan_unique_TRAIN_requests':48,'plan_exposures_each':8,'plan_exposures':384,'native_original_TRAIN_decisions':443,'native_experts':398,'native_certified_errors':45,'total_rows':827,'planned_optimizer_steps_each_arm':207,'chat_prompt_matches_frozen_runtime':True,'prompt_labels_all_minus100':True,'completion_only_no_truncation':True,'max_full_tokens':max(lengths),'total_completion_tokens':sum(completion_lengths),'confirmation_families':families,'fresh_full_sequence_overlap':0,'CPU_expert_gold_passed':32,'fresh_labels_not_in_training':True,'source_anchor_rows_unchanged':True,'model_predictions':0,'learning_steps':0,'sha256':{str(f):sha(f) for f in [D/'plan-supervision.jsonl',D/'mixed-supervision.jsonl',R/'confirmation-tasks.json',R/'confirmation-gold.jsonl',Path('scripts/prepare_round034_supervision.py'),Path('scripts/round034_core.py')]}});print({'TRAIN_rows':827,'prospective_families':families,'expert_gold32':True,'model_predictions':0})
if __name__=='__main__':main()
