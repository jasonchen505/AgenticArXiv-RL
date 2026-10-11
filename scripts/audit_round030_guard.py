#!/usr/bin/env python3
"""CPU TRAIN-only prototype checks; no heldout evaluation or changes to native agent."""
from pathlib import Path
import os,sys,json,datetime,hashlib,copy,collections,ast
os.environ.update(CUDA_VISIBLE_DEVICES='',STORE_BACKEND='memory',FIGURE_ANALYSIS_BACKEND='extractive');sys.path[:0]=['AgenticArxiv','scripts']
from request_execution_guard import RequestExecutionGuard,UnsupportedRequest,compile_request
R=Path('artifacts/reproduction/round-030-2026-10-09');OLD=R.parent/'round-024-2026-10-07';read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def main():
 assert not (R/'guard-preparation-plan.json').exists();tasks=read(OLD/'train-tasks.json');snapshot=Path('data/repro_round024/trainA-snapshot.json');assert len(tasks)==64
 source=Path('scripts/request_execution_guard.py');tree=ast.parse(source.read_text());assert not any(isinstance(x,ast.Name) and x.id in ['expected_tools','expected_tool_args','task_id','oracle','provenance'] for x in ast.walk(tree))
 write(R/'guard-preparation-plan.json',{'registered_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_only':True,'no_model_prediction_or_learning':True,'input':'user_instruction str only; no task metadata/oracle accepted by runtime module','development_cases':'Existing64 TRAIN only. No Dev/confirmation instructions or model failures used for selecting grammar or patches. Grammar covers explicit user workflow template; unsupported freeform abstains.','prototype':'Validate proposals before tool execution, commit actual successful observations with selectedpaper identity, reject premature FINISH; never auto-select/execute tools or replace model policy.','checks':['Compile all64TRAIN instructions from text, equal their declared tasks as audit targets only','CPU expert actions execute real snapshot tool environment and are admitted','Unknown tools/wrongarguments/prematureFINISH rejected','Error/wrongpaper observations do not advance state','Ready FINISH accepted once; afterFINISH and replayedtools rejected','Unsupported/non-text input abstains'],'future_gate':'Prototype CPU checks are not Agent/model gains. Before any model comparison, separately register bounded retry/compute budgets, native fallback, complete proposal+execution traces and metrics including denied proposals; do not silently remove failures or use oracle prompt.','sha256':{str(p):sha(p) for p in [source,Path(__file__),OLD/'train-tasks.json',snapshot]}})
 from tools.bootstrap import require_all_tools
 from rl.multiturn_env import AgenticArxivMultiTurnEnv
 from benchmark.semantic_oracle import _call
 require_all_tools('R030 TRAIN-only execution guard');results=[];checks=collections.Counter()
 import tempfile
 with tempfile.TemporaryDirectory(prefix='r030-guard-') as tmp:
  for k,n in [('PDF_RAW_PATH','raw'),('PDF_TRANSLATED_PATH','translated'),('PDF_FIGURES_PATH','figures')]:os.environ[k]=str(Path(tmp)/n)
  for t in tasks:
   guard=RequestExecutionGuard(t['task']);assert [s.name for s in guard.steps]==t['expected_tools'] and [s.args for s in guard.steps]==t['expected_tool_args'];checks['instruction_compilations']+=1;env=AgenticArxivMultiTurnEnv(snapshot);env.reset(task_id=t['id'])
   for name,args in zip(t['expected_tools'],t['expected_tool_args']):
    action={'name':name,'args':args};before=copy.deepcopy(guard.__dict__);assert not guard.check('FINISH').allowed;assert guard.__dict__==before;checks['premature_FINISH_rejections']+=1
    assert not guard.check({'name':'describe','args':args}).allowed;assert guard.__dict__==before;checks['unknown_tool_rejections']+=1
    wrong=copy.deepcopy(action);key=next(iter(wrong['args']));wrong['args'][key]='__invalid__';assert not guard.check(wrong).allowed;assert guard.__dict__==before;checks['wrong_argument_rejections']+=1
    assert guard.check(action).allowed;assert not guard.observe(action,{'error':'controlled tool failure'}).allowed;assert guard.__dict__==before;checks['failed_observation_state_unchanged']+=1
    if name!='get_recently_submitted_cs_papers':assert not guard.observe(action,{'paper_id':'__wrong__','status':'READY','content':'x','summary':'x','count':1,'answer':'x'}).allowed;assert guard.__dict__==before;checks['wrong_paper_state_unchanged']+=1
    result=_call(env,name,args);decision=guard.observe(action,result);assert decision.allowed,(t['id'],name,result,decision);checks['successful_TOOL_commits']+=1
   assert guard.check('FINISH').allowed;assert guard.observe('FINISH',None).allowed;assert not guard.check('FINISH').allowed;checks['completed_FINISH_and_duplicate_rejection']+=1;results.append({'task_id':t['id'],'steps':len(guard.steps),'finished':True})
 for text in ['找一篇论文',{'expected_tools':['download_arxiv_pdf']},'检索最近7天人工智能(cs.AI)论文5篇，下载第1篇，然后按以下顺序处理同一篇论文：描述它的第1张图。完成这些步骤后结束。']:
  try:compile_request(text)
  except UnsupportedRequest:checks['unsupported_abstentions']+=1
  else:raise AssertionError('Unsupported case accepted')
 write(R/'guard-prototype-audit.json',{'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'CPU_only':True,'TRAIN_only':True,'nominal_tasks':64,'visible_contracts':48,'counts':dict(checks),'results':results,'oracle_not_runtime_input':True,'native_agent_unchanged':True,'model_predictions':0,'new_learning_steps':0,'no_model_or_benchmark_gain_claim':True,'source_sha256':sha(source)});print(dict(checks))
if __name__=='__main__':main()
