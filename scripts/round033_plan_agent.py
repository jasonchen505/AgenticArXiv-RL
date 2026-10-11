"""One model-authored open-loop plan. No request compiler, oracle, or retries."""
import copy,json,time
from arm_blind_observations import ArmBlindAgent

def parse_plan(text):
 def unique(pairs):
  d={}
  for k,v in pairs:
   if k in d:raise ValueError('duplicate JSON key: '+k)
   d[k]=v
  return d
 def bad_constant(x):raise ValueError('non-finite JSON constant')
 p=json.loads(text,object_pairs_hook=unique,parse_constant=bad_constant)
 if not isinstance(p,dict) or set(p)!={'steps','final_thought'}:raise ValueError('exact top-level fields required')
 if not isinstance(p['final_thought'],str) or not p['final_thought'].strip():raise ValueError('nonempty final_thought required')
 if not isinstance(p['steps'],list) or not 0<=len(p['steps'])<=8:raise ValueError('0..8 planned steps required')
 for a in p['steps']:
  if not isinstance(a,dict) or set(a)!={'name','args'} or not isinstance(a['name'],str) or not a['name'] or not isinstance(a['args'],dict):raise ValueError('each step requires name:string and args:object')
 return p

class PlanAgent(ArmBlindAgent):
 stop_sequences=()
 def __init__(self,llm_client,**kwargs):super().__init__(llm_client,mode='native',**kwargs)
 def _dispatch_tool(self,name,args):
  try:return super()._dispatch_tool(name,args)
  except Exception as e:
   self.tool_dispatch_errors.append({'tool':name,'args':copy.deepcopy(args),'exception':str(e)});raise
 def plan_request(self,task,agent_model=None,initial_history=''):
  tools=self.format_tools_for_prompt(self.discover_tools())
  prompt=('你是论文工具助手。根据用户请求，一次性制定完整且按顺序执行的工具计划。只输出一个合法 JSON 对象，不要 Markdown、解释文字或 Observation。'
   '精确格式为 {"steps":[{"name":"工具名","args":{}}],"final_thought":"完成计划后的简短说明"}。'
   'steps 最多8项，保留用户明确要求的工具顺序及参数；工具名与参数必须遵守以下工具说明。'
   '可以用工具的 ref 参数引用搜索结果序号；不要猜测尚未获取的论文ID、路径或工具结果。'
   '计划将按顺序实际执行，任一工具失败就停止，不会补写或修复你的计划。\n\n可用工具：\n'+tools+'\n\n用户请求：\n'+task)
  if initial_history.strip():prompt+='\n\n已发生的会话上下文：\n'+initial_history.strip()
  return dict(model=agent_model,messages=[{'role':'user','content':prompt}],temperature=.1,max_tokens=1000,stream=False,extra=None)
 def generate(self,task,agent_model=None,initial_history=''):
  self.recorder.records=[];t=time.monotonic();response=self.llm_client.chat_completions(**self.plan_request(task,agent_model,initial_history));return copy.deepcopy(self.recorder.records[0]),int((time.monotonic()-t)*1000)
 def execute_prediction(self,task,prediction,llm_ms=0,session_id='default'):
  started=time.monotonic();self.session_id=session_id;self.artifact_observations=[];self.tool_dispatch_errors=[];self.raw_tool_result=None;history=[];executed=[];timings=[];events=[];p=None;error=None
  response=prediction.get('response',{});text=((response.get('choices') or [{}])[0].get('message') or {}).get('content','');usage=copy.deepcopy(response.get('usage') or {'prompt_tokens':0,'completion_tokens':0,'total_tokens':0})
  try:p=parse_plan(text)
  except Exception as e:
   error=str(e);history=[{'thought':'','action':text,'observation':'计划解析失败: '+error},{'thought':'','action':'ERROR','observation':'计划解析失败: '+error}];executed=[copy.deepcopy(history[-1])]
  if p is not None:
   failed=False;known={x['name'] for x in self.discover_tools()}
   for i,a in enumerate(p['steps']):
    tool_ms=0;actual=False
    if failed:observation='工具执行失败: 先前步骤失败，本模型提案未执行'
    else:
     t=time.monotonic();self.raw_tool_result=None;observation=self._execute_with_side_effects(copy.deepcopy(a));tool_ms=int((time.monotonic()-t)*1000);actual=True
     failed=a['name'] not in known or self.raw_tool_result is None or (isinstance(self.raw_tool_result,dict) and 'error' in self.raw_tool_result) or any(k in observation for k in ['工具执行失败:','错误: 工具','工具返回结果格式异常:']);error=observation if failed else error
    row={'thought':'','action':json.dumps(a,ensure_ascii=False),'observation':observation};history.append(row);timings.append({'llm_ms':0,'tool_ms':tool_ms});events.append({'plan_index':i,'action':copy.deepcopy(a),'attempted':actual,'failed_or_skipped':failed})
    if actual:executed.append(copy.deepcopy(row))
   terminal={'thought':p['final_thought'],'action':'ERROR' if failed else 'FINISH','observation':'计划执行失败，停止' if failed else '计划中的工具全部执行成功'};history.append(terminal);executed.append(copy.deepcopy(terminal))
  tool_ms=sum(x['tool_ms'] for x in timings);total_ms=llm_ms+int((time.monotonic()-started)*1000);timings.insert(0,{'llm_ms':llm_ms,'tool_ms':0})
  return {'task':task,'history':history,'execution_history':executed,'final_observation':history[-1]['observation'],'total_time_ms':total_ms,'iteration_count':len(history),'agent_type':'single_plan','timing':{'total_llm_ms':llm_ms,'total_tool_ms':tool_ms,'framework_overhead_ms':total_ms-llm_ms-tool_ms,'steps':timings},'token_usage':usage,'proposal_trace':[copy.deepcopy(prediction)],'LLM_calls':1,'max_LLM_calls':1,'parsed_plan':p,'plan_parse_valid':p is not None,'plan_error':error,'plan_events':events,'artifact_observations':copy.deepcopy(self.artifact_observations),'tool_dispatch_errors':copy.deepcopy(self.tool_dispatch_errors),'stable_public_artifact_addresses':True,'finish_is_executor_all_planned_steps_succeeded':True,'model_final_thought_generated_before_observations':True}
