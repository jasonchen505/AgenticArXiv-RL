import os,sys,json,itertools,collections,datetime
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',STORE_BACKEND='memory',FIGURE_ANALYSIS_BACKEND='extractive');sys.path[:0]=['AgenticArxiv','scripts']
from round036_core import R,D,read,rows,write,sha,encode
for k,n in [('PDF_RAW_PATH','raw'),('PDF_TRANSLATED_PATH','translated'),('PDF_FIGURES_PATH','figures')]:os.environ[k]=str((D/'CPU'/n).resolve())
from round033_plan_agent import PlanAgent,parse_plan
from round035_literal_error_log import install
from tools.bootstrap import require_all_tools
from agents.prompt_templates import build_visible_setup_context
from benchmark.runner import BenchmarkRunner
from benchmark.semantic_oracle import attach_expected_paper_ids
from benchmark.metrics import extract_metrics,is_strict_success
from prepare_round033_plan import Fake
from transformers import AutoTokenizer
N={'C':'get_paper_content','S':'summarize_paper','F':'extract_paper_figures','A':'analyze_figure'}
def main():
 assert read(R/'closure-independent-audit.json')['fixed_recipe_closed'];assert not (R/'curriculum-CPU-audit.json').exists();D.mkdir(exist_ok=True);(D/'tmp').mkdir(exist_ok=True)
 families=sorted(''.join(x) for x in itertools.permutations('CSFA') if x.index('F')<x.index('A'));reserved=['SFAC','SFCA'];trainfamilies=[x for x in families if x not in reserved];assert len(trainfamilies)==10
 write(R/'curriculum-selection.json',{'registered_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'TRAIN_full_families':trainfamilies,'reserved_unmeasured_full_families':reserved,'reserved_task_texts_labels_execution_or_model_predictions_created':False,'selection_before_new_labels_gold_or_model':True,'old_FSAC_FSCA_now_seen_diagnostic_not_fresh':True,'hypothesis':'Expanded six-step TRAIN coverage plus separate plan adapter; two changes assessed jointly, not a component-causal claim','new_stage_training_uses_known_family_grammar_only':True})
 original=read(R.parent/'round-033-2026-10-10/TRAIN-unique-tasks.json');new=[]
 for f,(aspect,ref,words) in itertools.product(trainfamilies,itertools.product(['AI','CV'],[1,3],[60,120])):
  style='tldr' if words==60 else 'structured';spec={'C':{'ref':ref,'section':None},'S':{'ref':ref,'style':style,'max_words':words},'F':{'ref':ref},'A':{'ref':ref,'figure_no':1,'question':'describe'}};phrases={'C':'读取它默认返回的标题与摘要','S':f'以{style}风格用不超过{words}词总结它','F':'抽取它的全部图表','A':'描述它的第1张图'}
  new.append({'id':f'plan036_{f}_{aspect}_{ref}_{words}','task':'检索最近7天'+('人工智能(cs.AI)' if aspect=='AI' else '计算机视觉(cs.CV)')+f'论文5篇，下载第{ref}篇，然后按以下顺序处理同一篇论文：'+'；'.join(phrases[c] for c in f)+'。完成这些步骤后结束。','expected_tools':['get_recently_submitted_cs_papers','download_arxiv_pdf']+[N[c] for c in f],'expected_tool_args':[{'aspect':aspect,'days':7,'max_results':5},{'ref':ref}]+[spec[c] for c in f],'expected_termination':'FINISH','category':'TRAIN_long_coverage','max_iterations':8,'requires_offline':True,'provenance':{'family':f,'TRAIN_schema_synthetic_not_model_distillation':True}})
 tasks=original+new;assert len(tasks)==128 and len({t['task'] for t in tasks})==128;write(R/'TRAIN-tasks.json',tasks)
 previous=read(R.parent/'round-034-2026-10-10/experiment-plan.json');devtexts={t['task'] for g in ['olddev','newdev'] for t in read(previous['eval_jobs'][f'source-native-{g}-s49']['tasks'])};assert not devtexts.intersection(t['task'] for t in tasks)
 tok=AutoTokenizer.from_pretrained(previous['source_model'],local_files_only=True);a=PlanAgent(None);data=[];lengths=[];lengthcounts=collections.Counter()
 for t in tasks:
  prompt=a.plan_request(t['task'],previous['source_model'],build_visible_setup_context(t))['messages'];target={'steps':[{'name':n,'args':v} for n,v in zip(t['expected_tools'],t['expected_tool_args'])],'final_thought':'请求中的工具步骤已完成'};text=json.dumps(target,ensure_ascii=False,separators=(',',':'));assert parse_plan(text)==target;assert not any(x in json.dumps(prompt,ensure_ascii=False) for x in ['expected_tools','expected_tool_args','provenance',t['id'],'repro_round036']);row={'kind':'plan','source_task_id':t['id'],'prompt':prompt,'chosen':[{'role':'assistant','content':text}]};z=encode(tok,row);lengths.append(len(z['input_ids']));lengthcounts[len(target['steps'])]+=1;data.append(row)
 # This stage intentionally learns known full families. No reserved program text or labels are materialized.
 repeated=[dict(x,exposure=i) for i in range(4) for x in data]
 for name,xs in [('plan-supervision.jsonl',data),('training-supervision.jsonl',repeated)]:assert not (D/name).exists();(D/name).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in xs))
 for k,n in [('PDF_RAW_PATH','raw'),('PDF_TRANSLATED_PATH','translated'),('PDF_FIGURES_PATH','figures')]:os.environ[k]=str((D/'CPU'/n).resolve())
 install();require_all_tools('R036 TRAIN curriculum expert executability');gold=[]
 for world in ['trainA','trainB']:
  snapshot='data/repro_round024/'+world+'-snapshot.json';runner=BenchmarkRunner(agent_types=['regex'],offline=True,snapshot=snapshot)
  for t in attach_expected_paper_ids(tasks,snapshot):
   runner._reset_offline_trial_state();target={'steps':[{'name':n,'args':v} for n,v in zip(t['expected_tools'],t['expected_tool_args'])],'final_thought':'请求中的工具步骤已完成'};agent=PlanAgent(Fake(json.dumps(target)),side_effect_mgr=runner._side_effects(),env=runner._tool_env());q,ms=agent.generate(t['task']);z=agent.execute_prediction(t['task'],q,session_id='CPU_'+t['id']);m=extract_metrics(t,z,'CPU',0,'CPU_'+t['id']);assert is_strict_success(m),t['id'];gold.append({'world':world,'task_id':t['id'],'history':z['history'],'metrics':m.to_dict()})
 (R/'TRAIN-expert-gold.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in gold));write(R/'curriculum-CPU-audit.json',{'passed':True,'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'unique_TRAIN_requests':128,'old_short48':48,'new_long80':80,'full_tool_length_counts':dict(lengthcounts),'content_and_summary_cooccur_requests':80,'exposures_each':4,'total_training_rows':512,'native_training_rows':0,'TRAIN_Dev_request_overlap':0,'completion_only_prompt_mask_minus100_runtime_token_prefix_exact':True,'truncation':0,'max_full_tokens':max(lengths),'expert_CPU256_strict':256,'expert_gold_is_not_model_success':True,'reserved_families':reserved,'reserved_materialized':False,'new_model_predictions':0,'new_learning_steps':0,'sha256':{str(f):sha(f) for f in [D/'training-supervision.jsonl',D/'plan-supervision.jsonl',R/'TRAIN-tasks.json',R/'TRAIN-expert-gold.jsonl',R/'curriculum-selection.json']}});print(read(R/'curriculum-CPU-audit.json'))
if __name__=='__main__':main()
