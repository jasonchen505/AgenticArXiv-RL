#!/usr/bin/env python3
from pathlib import Path
import os,sys,json,argparse,time,hashlib
sys.path[:0]=['AgenticArxiv','scripts']
R=Path('artifacts/reproduction/round-031-2026-10-09');read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
from benchmark.runner import BenchmarkRunner
from round031_guard_agent import ExecutionAwareAgent
class ExecutionRunner(BenchmarkRunner):
 def __init__(self,*args,mode,**kwargs):self.execution_mode=mode;super().__init__(*args,**kwargs)
 def _create_agent(self,agent_type,max_iterations=None):
  assert agent_type=='regex';base=max_iterations or 5;limit=base if self.execution_mode=='native' else base+4
  return ExecutionAwareAgent(self.llm_client,mode=self.execution_mode,side_effect_mgr=self._side_effects(),env=self._tool_env(),max_iterations=limit,llm_extra=self.llm_extra,prompt_policy='baseline')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',required=True);ap.add_argument('--group',required=True);ap.add_argument('--seed',type=int,required=True);a=ap.parse_args();p=read(R/'experiment-plan.json');assert a.mode in p['modes'] and a.group in p['groups'] and a.seed in p['generation_seeds']
 for f,h in p['sha256'].items():assert sha(f)==h,f
 for f,h in p['source_model_freeze']['sha256'].items():assert sha(f)==h,f
 g=p['groups'][a.group];sf,key=g['split'].rsplit(':',1);ids=read(sf)['split'][key];tasks0=read(g['tasks']);tasks=[{t['id']:t for t in tasks0}[tid] for tid in sorted(ids)];out=R/(a.group+'-'+a.mode+'-s'+str(a.seed));assert not out.exists();out.mkdir()
 from tools.bootstrap import require_all_tools
 from benchmark.metrics import extract_metrics,is_strict_success
 from benchmark.report import BenchmarkReport
 from benchmark.semantic_oracle import attach_expected_paper_ids
 require_all_tools('Frozen execution architecture');runner=ExecutionRunner(mode=a.mode,agent_types=['regex'],repeat=3,model=p['source_model'],session_prefix='bench_r20261003_075529',offline=True,snapshot=g['snapshot'],llm_backend='transformers',local_device='cuda',local_dtype='bfloat16',generation_seed=a.seed,prompt_policy='baseline');results=runner.run_all(tasks);oracles={t['id']:t for t in attach_expected_paper_ids(tasks,g['snapshot'])};traces=[];extra=[]
 for result in results:
  raw=result.raw_result;assert not result.error and result.metrics is not None;assert raw['LLM_calls']<=raw['max_LLM_calls'];assert len(raw['proposal_trace'])==raw['LLM_calls'];executed=extract_metrics(oracles[result.task_id],dict(raw,history=raw['execution_history']),'executed_regex',result.trial,result.session_id);traces.append({'task_id':result.task_id,'trial':result.trial,'session_id':result.session_id,'history':raw['history'],'raw_result':raw});extra.append({'task_id':result.task_id,'trial':result.trial,'proposal_strict':is_strict_success(result.metrics),'proposal_FF':result.metrics.false_finish,'execution_strict':is_strict_success(executed),'execution_FF':executed.false_finish,'execution_metrics':executed.to_dict(),'LLM_calls':raw['LLM_calls'],'max_LLM_calls':raw['max_LLM_calls'],'vetoes':raw['guard_vetoes'],'contract_supported':raw['contract_supported'],'actual_tool_attempts':sum(h['action'] not in ['FINISH','FORCE_STOP','ERROR'] for h in raw['execution_history']),'total_tokens':raw['token_usage']['total_tokens']})
 assert len(traces)==len(tasks)*3;BenchmarkReport([r.metrics for r in results],model=p['source_model'],errors=[]).save_all(str(out));(out/'traces.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False,default=str)+'\n' for x in traces));(out/'execution-summary.json').write_text(json.dumps({'runs':len(extra),'details':extra},ensure_ascii=False,indent=2)+'\n');(out/'execution_protocol.json').write_text(json.dumps({'mode':a.mode,'group':a.group,'seed':a.seed,'repeat':3,'weights_updated':False,'source_model':p['source_model'],'input_user_instruction_only':True,'oracle_in_runtime_guard_or_model_prompt':False,'original_proposal_scorer_unchanged':True,'execution_metrics_separate_not_replacement':True,'max_completion_tokens_per_call':1000,'temperature':.1,'dtype':'bfloat16','limit_policy':'native requestedmax/default5;allotherarms+4','full_proposals_and_denials_preserved':True,'tasks_sha256':sha(g['tasks'])},indent=2)+'\n');print('Completed',len(traces),'full proposal and execution trials')
if __name__=='__main__':main()
