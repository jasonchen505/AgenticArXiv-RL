import time
from .native import call
from .protocol import state_actions,canonical_units,selection_prompt as old_prompt,parse_selection,render,planning_prompt,parse_plan,status_from_events
from .identity import resolve
from .retrieval import BodyEvidenceEnv as PaperEvidenceEnv
def run(client,a,p,bank,cat,t):
 arm=a['native'];client.task_id=t['id'];start=time.monotonic();calls=[];errors=[];raw=None;cert=None
 plan,cert=resolve(t['task'],cat) if a['mode']=='compiled' else (None,None)
 fallback=a['mode']=='compiled' and plan is None
 if a['mode']=='two_call' or fallback:
  raw,err=call(client,arm,'paper_only_plan',planning_prompt(t['task'],cat),calls)
  try:
   if err:raise ValueError(err)
   plan=parse_plan(raw,cat)
  except Exception as e:errors.append({'phase':'paper_only_plan','error':str(e)})
 env=PaperEvidenceEnv(bank,t['id'] in p['fault_ids']);stop='invalid_plan'
 if plan:
  stop='tool_budget'
  for i in range(8):
   acts,arrived,_=state_actions(plan,env,cat,t['task'])
   if set(plan['paper_ids'])<=arrived:stop='planned_evidence_arrived';break
   if not acts:stop='no_legal_action';break
   env.execute(acts[0]['name'],acts[0]['args'])
  else:
   if set(plan['paper_ids'])<={c['paper_id'] for c in env.observed.values()}:stop='planned_evidence_arrived'
 units=canonical_units(env,plan or {'paper_ids':[]});prompt=old_prompt
 ans,err=call(client,arm,'original_task_sentence_selection',prompt(t['task'],units),calls);selection=None;answer=''
 try:
  if err:raise ValueError(err)
  selection=parse_selection(ans,units);text=render(selection,units,env);prefix='取证执行记录（来自实际工具事件，不是论文主张）：';assert text.count(prefix)==1
  answer='\n\n'.join(prefix+status_from_events(env.events) if part.startswith(prefix) else part for part in text.split('\n\n'))
 except Exception as e:errors.append({'phase':'original_task_sentence_selection','error':str(e)})
 return {'task_id':t['id'],'task':t['task'],'split':t['split'],'experiment_arm':a['name'],'selector':a['selector'],'mode':a['mode'],'arm':arm['arm'],'model':arm['model'],'thinking':arm['thinking'],'plan':plan,'plan_raw':raw,'identity_certificate':cert,'identity_LLM_calls':int(a['mode']=='two_call' or fallback),'routing_fallback':fallback,'answer_raw':ans,'selection':selection,'answer':answer,'units':units,'observed_evidence':env.observed,'events':env.events,'calls':calls,'LLM_calls':len(calls),'tool_calls':len(env.events),'errors':errors,'stop_reason':stop,'failure_injected':env.fired,'cache_recovery':env.fired and any(e['name']=='read_cached_paper' and e['success'] for e in env.events),'recovery_policy':'deterministic_executor','total_ms':int((time.monotonic()-start)*1000),'token_usage':{k:sum(c['response']['usage'][k] for c in calls) for k in ['prompt_tokens','completion_tokens','total_tokens','reasoning_tokens','final_visible_tokens']},'new_optimizer_steps':0,'no_private_gold_in_inputs':True,'common_candidate_status_renderer':True}
