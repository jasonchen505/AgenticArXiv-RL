#!/usr/bin/env python3
import os,sys,json,copy,tempfile,datetime
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',STORE_BACKEND='memory',FIGURE_ANALYSIS_BACKEND='extractive');sys.path[:0]=['AgenticArxiv','scripts'];R=Path('artifacts/reproduction/round-031-2026-10-09');read=lambda p:json.loads(Path(p).read_text())
from tools.bootstrap import require_all_tools
from benchmark.runner import BenchmarkRunner
from benchmark.metrics import extract_metrics,is_strict_success
from agents.agent_engine import ReActAgent
from round031_guard_agent import ExecutionAwareAgent
from agents.prompt_templates import build_visible_setup_context
class Fake:
 def __init__(self,answers):self.answers=iter(answers);self.requests=[]
 def chat_completions(self,**kwargs):
  self.requests.append(copy.deepcopy(kwargs));return {'choices':[{'message':{'content':next(self.answers)}}],'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}}
def run(t,mode,answers,limit=None):
 runner=BenchmarkRunner(agent_types=['regex'],offline=True,snapshot='data/repro_round024/trainA-snapshot.json');runner._reset_offline_trial_state();fake=Fake(answers);kwargs={'env':runner._tool_env(),'side_effect_mgr':runner._side_effects(),'max_iterations':limit or t['max_iterations']+4};agent=ReActAgent(fake,**kwargs) if mode=='original' else ExecutionAwareAgent(fake,mode=mode,**kwargs);raw=agent.run(t['task'],session_id='CPU_TRAIN_'+t['id'],initial_history=build_visible_setup_context(t));return raw,fake
require_all_tools('Round031 TRAIN-only native integration');tasks=read('artifacts/reproduction/round-024-2026-10-07/train-tasks.json');counts={};cases=[]
with tempfile.TemporaryDirectory(dir='data/repro_round031',prefix='cpu-') as tmp:
 for k,n in [('PDF_RAW_PATH','raw'),('PDF_TRANSLATED_PATH','translated'),('PDF_FIGURES_PATH','figures')]:os.environ[k]=str(Path(tmp)/n)
 for t in tasks:
  answers=['Thought: 按用户请求执行\nAction: '+json.dumps({'name':n,'args':a},ensure_ascii=False) for n,a in zip(t['expected_tools'],t['expected_tool_args'])]+['Thought: 所有步骤均已成功执行，任务已完成\nAction: FINISH']
  original,f0=run(t,'original',answers);native,f1=run(t,'budget',answers);assert native['history']==original['history'] and f0.requests==f1.requests;assert is_strict_success(extract_metrics(t,native,'CPU',0));counts['native_wrapper_exact_golden_history_and_requests']=counts.get('native_wrapper_exact_golden_history_and_requests',0)+1
  for mode in ['state','guard']:
   raw,_=run(t,mode,answers);assert is_strict_success(extract_metrics(t,raw,'CPU',0)) and is_strict_success(extract_metrics(t,dict(raw,history=raw['execution_history']),'CPU',0));assert raw['guard_vetoes']==0
   counts['gold_'+mode]=counts.get('gold_'+mode,0)+1
  bad='Thought: 尝试未知工具\nAction: '+json.dumps({'name':'describe','args':{'ref':1}});raw,_=run(t,'guard',[bad]+answers);assert raw['guard_vetoes']==1 and not is_strict_success(extract_metrics(t,raw,'CPU',0)) and is_strict_success(extract_metrics(t,dict(raw,history=raw['execution_history']),'CPU',0));assert len(raw['proposal_trace'])==len(answers)+1 and raw['history'][0]['action'].find('describe')>=0;counts['denied_proposal_preserved_original_strict_fails_execution_strict_passes']=counts.get('denied_proposal_preserved_original_strict_fails_execution_strict_passes',0)+1
  early='Thought: 已完成\nAction: FINISH';raw,_=run(t,'guard',[early]+answers);assert raw['guard_vetoes']==1 and not is_strict_success(extract_metrics(t,raw,'CPU',0));assert is_strict_success(extract_metrics(t,dict(raw,history=raw['execution_history']),'CPU',0));counts['earlyFINISH_consumes_call_and_preserved']=counts.get('earlyFINISH_consumes_call_and_preserved',0)+1
  cases.append(t['id'])
 t=copy.deepcopy(tasks[0]);t['task']='一个不支持的自由文本请求';answers=['Thought: 完成\nAction: FINISH'];orig,f0=run(t,'original',answers);raw,f1=run(t,'guard',answers);assert not raw['contract_supported'] and raw['history']==orig['history'] and f0.requests==f1.requests;counts['unsupported_native_fallback_exact']=1
 t=tasks[0];bad='Thought: 未完成\nAction: '+json.dumps({'name':'describe','args':{}});raw,_=run(t,'guard',[bad]*4,limit=4);assert raw['LLM_calls']==raw['guard_vetoes']==4 and raw['history'][-1]['action']=='FORCE_STOP' and raw['execution_history'][-1]['action']=='FORCE_STOP';counts['budget_exhaustion_no_falseFINISH']=1
(R/'integration-cpu-audit.json').write_text(json.dumps({'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'TRAIN_only':True,'CPU_only':True,'model_predictions':0,'learning_steps':0,'counts':counts,'tasks':cases,'scripted_expert_not_model_gain':True,'full_denied_proposals_retained':True,'native_source_not_edited':True},indent=2)+'\n');print(counts)
