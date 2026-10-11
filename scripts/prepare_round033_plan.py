"""TRAIN-only CPU preparation; expert JSON is a fixture, not model evidence."""
import os,sys,json,copy,datetime,tempfile,collections
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',STORE_BACKEND='memory',FIGURE_ANALYSIS_BACKEND='extractive')
sys.path[:0]=['AgenticArxiv','scripts']
from round031_study_core import read,write,sha
from round033_plan_agent import PlanAgent,parse_plan
from tools.bootstrap import require_all_tools
from tools.tool_registry import registry
from benchmark.runner import BenchmarkRunner
from benchmark.metrics import extract_metrics,is_strict_success
from benchmark.semantic_oracle import attach_expected_paper_ids
from agents.prompt_templates import build_visible_setup_context
R=Path('artifacts/reproduction/round-033-2026-10-10')
class Fake:
 def __init__(self,text):self.text=text
 def chat_completions(self,**kw):return {'choices':[{'message':{'content':self.text},'finish_reason':'stop'}],'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}}
def main():
 require_all_tools('R033 plan CPU preparation');assert read(R/'parity-independent-audit.json')['passed'];assert not (R/'plan-CPU-audit.json').exists();tasks=sorted(read('artifacts/reproduction/round-024-2026-10-07/train-tasks.json'),key=lambda x:x['id']);unique={};aliases={}
 for t in tasks:
  assert not t.get('setup') and not t.get('depends_on');visible=json.dumps([t['task'],build_visible_setup_context(t)],ensure_ascii=False);chosen=unique.setdefault(visible,t);aliases[t['id']]=chosen['id']
 selected=sorted(unique.values(),key=lambda x:x['id']);assert len(selected)==48
 write(R/'TRAIN-unique-tasks.json',selected);write(R/'TRAIN-aliases.json',{'selection':'first lexicographic ID for each (user request, visible setup context), before model outputs','nominal_tasks':64,'unique_visible':48,'aliases':aliases,'family_counts':dict(collections.Counter(t['provenance']['family'] for t in selected))})
 passed=0;steps=0;requests={};root=Path('data/repro_round033/CPU').resolve();root.mkdir(parents=True,exist_ok=True)
 for k,n in [('PDF_RAW_PATH','raw'),('PDF_TRANSLATED_PATH','translated'),('PDF_FIGURES_PATH','figures')]:os.environ[k]=str(root/n)
 for world in ['trainA','trainB']:
  snapshot='data/repro_round024/'+world+'-snapshot.json';runner=BenchmarkRunner(agent_types=['regex'],offline=True,snapshot=snapshot)
  for t in attach_expected_paper_ids(selected,snapshot):
   runner._reset_offline_trial_state();p={'steps':[{'name':n,'args':a} for n,a in zip(t['expected_tools'],t['expected_tool_args'])],'final_thought':'请求中的步骤全部完成'};agent=PlanAgent(Fake(json.dumps(p,ensure_ascii=False)),env=runner._tool_env(),side_effect_mgr=runner._side_effects());pred,ms=agent.generate(t['task']);z=agent.execute_prediction(t['task'],pred,session_id='CPU_'+t['id']);assert is_strict_success(extract_metrics(t,z,'CPU',0,'CPU_'+t['id']));assert z['execution_history']==z['history'];assert len(z['artifact_observations'])==len(p['steps']);passed+=1;steps+=len(p['steps']);key=t['id']
   if world=='trainA':requests[key]=pred['request']
   else:assert requests[key]==pred['request']
   assert not any(s in json.dumps(pred['request'],ensure_ascii=False) for s in ['expected_tools','expected_tool_args','provenance','data/repro_round033',t['id']]);assert agent.discover_tools()==registry.list_tools()
 # Strict parsing and retention checks: no salvage, oracle repair, or automatic tool substitution.
 invalid=['```json\n{}\n```','{}','{"steps":[],"steps":[],"final_thought":"x"}','{"steps":[],"final_thought":NaN}',json.dumps({'steps':[{'name':'x','args':{}}]*9,'final_thought':'x'}),json.dumps({'steps':[{'name':'x','args':[]}],'final_thought':'x'}),json.dumps({'steps':[],'final_thought':''})]
 for text in invalid:
  try:parse_plan(text)
  except (ValueError,TypeError):pass
  else:raise AssertionError('invalid JSON plan accepted')
  a=PlanAgent(Fake(text),env=runner._tool_env(),side_effect_mgr=runner._side_effects());pred,_=a.generate('测试');z=a.execute_prediction('测试',pred);assert not z['plan_parse_valid'] and z['history'][0]['action']==text and z['history'][-1]['action']=='ERROR' and not z['artifact_observations']
 p={'steps':[{'name':'unknown_TOOL','args':{}},{'name':'download_arxiv_pdf','args':{'ref':1}}],'final_thought':'已完成'};a=PlanAgent(Fake(json.dumps(p)),env=runner._tool_env(),side_effect_mgr=runner._side_effects());pred,_=a.generate('测试');z=a.execute_prediction('测试',pred);assert z['plan_parse_valid'] and len(z['history'])==3 and len(z['execution_history'])==2 and z['history'][-1]['action']=='ERROR' and '未执行' in z['history'][1]['observation'] and not z['artifact_observations']
 # A real offline tool failure must halt later model-authored proposals without dropping them.
 p={'steps':[{'name':'download_arxiv_pdf','args':{'ref':99}},{'name':'get_paper_content','args':{'ref':99}}],'final_thought':'已完成'};runner._reset_offline_trial_state();a=PlanAgent(Fake(json.dumps(p)),env=runner._tool_env(),side_effect_mgr=runner._side_effects());pred,_=a.generate('测试');z=a.execute_prediction('测试',pred);assert z['history'][-1]['action']=='ERROR' and len(z['history'])==3 and len(z['artifact_observations'])+len(z['tool_dispatch_errors'])==1 and len(z['execution_history'])==2 and not z['plan_events'][1]['attempted']
 write(R/'plan-CPU-audit.json',{'passed':True,'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'TRAIN_only':True,'nominal64_deduplicated_before_model':48,'gold_plan_executions_both_old_TRAIN_worlds':passed,'actual_native_tool_steps':steps,'full_plan_proposal_history_equals_execution_on_success':True,'invalid_strict_parser_cases':len(invalid),'unknown_tool_and_real_failure_stop_remaining_proposals_preserved':True,'all_native_registered_tool_descriptions':True,'native_tool_count':len(registry.list_tools()),'requests_identical_across_worlds':48,'runtime_only_request_tools_visible_context':True,'model_predictions':0,'learning_steps':0,'sha256':{str(f):sha(f) for f in [Path('scripts/round033_plan_agent.py'),Path('scripts/prepare_round033_plan.py'),R/'TRAIN-unique-tasks.json',R/'TRAIN-aliases.json']}});print({'CPU_expert_runs':passed,'native_steps':steps,'real_model_predictions':0})
if __name__=='__main__':main()
