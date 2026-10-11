"""Separate inference architecture; native agent/scorer files remain unchanged."""
import copy,json
from agents.agent_engine import ReActAgent
from request_execution_guard import RequestExecutionGuard,UnsupportedRequest
from round031_native_loop import guarded_native_loop

class RecordingClient:
    def __init__(self,client):self.client=client;self.records=[]
    def chat_completions(self,**kwargs):
        record={'request':copy.deepcopy(kwargs)};self.records.append(record)
        try:response=self.client.chat_completions(**kwargs)
        except Exception as e:record['error']=str(e);raise
        record['response']=copy.deepcopy(response);return response
    def __getattr__(self,name):return getattr(self.client,name)

class ExecutionAwareAgent(ReActAgent):
    def __init__(self,llm_client,mode,**kwargs):
        assert mode in ['native','budget','state','guard'];self.mode=mode;self.recorder=RecordingClient(llm_client);super().__init__(self.recorder,**kwargs)
    def run(self,task,agent_model=None,session_id='default',initial_history=''):
        self.guard=None;self.execution_history=[];self.events=[];self.raw_tool_result=None;self.recorder.records=[]
        if self.mode in ['state','guard']:
            try:self.guard=RequestExecutionGuard(task)
            except UnsupportedRequest:pass
        if self.guard is None:
            result=super().run(task,agent_model,session_id,initial_history)
            self.execution_history=copy.deepcopy(result['history'])
            for record in self.recorder.records:
                if 'response' in record:
                    thought,action=self.parse_response(record['response']);self.events.append({'thought':thought,'action':'FINISH' if action is None else copy.deepcopy(action),'vetoed':False,'guard_checked':False})
            return self._guard_output(result)
        return guarded_native_loop(self,task,agent_model,session_id,initial_history)
    def _state_task(self,task):
        state=self.guard.status();info={'completed_steps':state['completed_steps'],'requested_steps':state['requested_steps']}
        if state['ready_to_finish']:info['next_requested_action']='FINISH'
        else:
            step=self.guard.steps[self.guard.position];info['next_requested_action']={'name':step.name,'args':dict(step.args)}
        return task+'\n\n[从本次用户指令及真实工具观察编译的状态]\n'+json.dumps(info,ensure_ascii=False)+'\n请按此用户请求的当前步骤提出动作；Observation仍必须来自真实工具。'
    def _guard_veto(self,thought,action,history,timings,llm_ms,iteration,msg_id):
        proposal='FINISH' if action is None else action;decision=self.guard.check(proposal);veto=self.mode=='guard' and not decision.allowed;self.raw_tool_result=None
        self.events.append({'thought':thought,'action':copy.deepcopy(proposal),'guard_checked':True,'contract_allowed':decision.allowed,'contract_reason':decision.reason,'vetoed':veto,'state_before':copy.deepcopy(self.guard.status())})
        if not veto:return False
        observation='工具执行失败: 执行校验拒绝（未执行）：'+decision.reason+'。请核对原用户请求后重新提出动作。'
        text=proposal if isinstance(proposal,str) else json.dumps(proposal,ensure_ascii=False);history.append({'thought':thought,'action':text,'observation':observation});timings.append({'llm_ms':llm_ms,'tool_ms':0});self._log_step(msg_id,iteration,thought,text,'',observation,llm_ms,0,self.session_id)
        if iteration==self.max_iterations-1:
            stop={'thought':'达到最大迭代次数','action':'FORCE_STOP','observation':'迭代限制'};history.append(stop);timings.append({'llm_ms':0,'tool_ms':0})
        return True
    def _dispatch_tool(self,name,args):
        result=super()._dispatch_tool(name,args);self.raw_tool_result=copy.deepcopy(result);return result
    def _guard_tool(self,thought,action,observation):
        row={'thought':thought,'action':json.dumps(action,ensure_ascii=False),'observation':observation};self.execution_history.append(row)
        result=self.raw_tool_result if self.raw_tool_result is not None else {'error':observation};decision=self.guard.observe(action,result);self.events[-1].update(observation_commit=decision.allowed,observation_commit_reason=decision.reason,state_after=copy.deepcopy(self.guard.status()),trusted_tool_result=copy.deepcopy(result))
    def _guard_terminal(self,thought,observation):
        self.execution_history.append({'thought':thought,'action':'FINISH','observation':observation});decision=self.guard.observe('FINISH',None);self.events[-1].update(observation_commit=decision.allowed,state_after=copy.deepcopy(self.guard.status()))
    def _guard_output(self,result):
        if result['history'] and result['history'][-1]['action'] in ['FORCE_STOP','ERROR']:
            tail=copy.deepcopy(result['history'][-1])
            if not self.execution_history or self.execution_history[-1]!=tail:self.execution_history.append(tail)
        assert len(self.recorder.records)<=self.max_iterations
        result.update(mode=self.mode,contract_supported=self.guard is not None,proposal_trace=[dict(copy.deepcopy(record),event=copy.deepcopy(self.events[i]) if i<len(self.events) else {'unparsed_error':True}) for i,record in enumerate(self.recorder.records)],execution_history=copy.deepcopy(self.execution_history),guard_vetoes=sum(x.get('vetoed',False) for x in self.events),LLM_calls=len(self.recorder.records),max_LLM_calls=self.max_iterations)
        return result
