import os,sys,json,datetime,collections,copy
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',STORE_BACKEND='memory',FIGURE_ANALYSIS_BACKEND='extractive');sys.path[:0]=['AgenticArxiv','scripts']
from round034_core import R,D,read,rows,sha,write,verify
S=R.parent/'round-035-2026-10-10'
from tools.bootstrap import require_all_tools
from benchmark.runner import BenchmarkRunner
from benchmark.semantic_oracle import attach_expected_paper_ids
from benchmark.metrics import extract_metrics,is_strict_success
from round033_plan_agent import PlanAgent
from round035_literal_error_log import install
def replay(label,p):
 spec=p['eval_jobs'][label];out=R/label;predictions=rows(out/'predictions.jsonl');assert len(predictions)==48 and read(out/'before-execution.json')['sha256']==sha(out/'predictions.jsonl');byid={(q['task_id'],q['trial']):q for q in predictions};tasks=read(spec['tasks']);assert set(byid)=={(t['id'],0) for t in tasks};rs=[]
 for k,n in [('PDF_RAW_PATH','raw'),('PDF_TRANSLATED_PATH','translated'),('PDF_FIGURES_PATH','figures')]:os.environ[k]=str((D/label/n).resolve())
 for world in ['trainA','trainB']:
  r=BenchmarkRunner(agent_types=['regex'],offline=True,snapshot=p['worlds'][world])
  for t in attach_expected_paper_ids(tasks,p['worlds'][world]):
   r._reset_offline_trial_state();session='bench_r20261003_075529_'+t['id']+'_regex_0';a=PlanAgent(None,side_effect_mgr=r._side_effects(),env=r._tool_env());q=byid[t['id'],0];z=a.execute_prediction(t['task'],copy.deepcopy(q['prediction']),q['llm_ms'],session);m=extract_metrics(t,z,'single_plan',0,session);em=extract_metrics(t,dict(z,history=z['execution_history']),'executed_single_plan',0,session);assert z['proposal_trace']==[q['prediction']]
   rs.append({'task_id':t['id'],'world':world,'trial':0,'session_id':session,'raw_result':z,'metrics':m.to_dict(),'execution_metrics':em.to_dict(),'strict':is_strict_success(m),'execution_strict':is_strict_success(em),'one_prediction_reused_two_worlds':True,'per_logical_run_full_prediction_cost':True})
 return rs
def main():
 amendment=read(S/'operation-amendment.json');verify(amendment);p=read(R/'experiment-plan.json');verify(p)
 for f,h in read(R/'all-finals-before-inference.json')['sha256'].items():assert sha(f)==h
 assert read(R/'mixed49-plan-TRAIN-s49-job.json')['exit_code']==1;require_all_tools('R035 CPU replay of original frozen plans, no prediction generation')
 # Reproduce the logger failure without executing any task, before installing the adapter.
 import agents.base_agent as base
 try:base.log.error('tool error key={"paper_id":"fixture"}',exc_info=True)
 except KeyError as e:write(S/'literal-log-original-failure.json',{'exception_type':'KeyError','message':str(e),'model_predictions':0})
 else:raise AssertionError('Expected original brace-formatting failure')
 log=install();log.error('tool error key={"paper_id":"fixture"}',exc_info=True);assert log.records==['tool error key={"paper_id":"fixture"}'];log.records=[]
 source=replay('source-plan-TRAIN-s49',p);original=rows(R/'source-plan-TRAIN-s49/traces.jsonl');assert len(source)==len(original)==96
 for x,y in zip(source,original):
  untimed=lambda z:{k:v for k,v in z.items() if k not in ['total_time_ms','total_tool_ms','framework_overhead_ms','avg_tool_ms']}
  assert untimed(x['metrics'])==untimed(y['metrics']) and untimed(x['execution_metrics'])==untimed(y['execution_metrics']) and x['strict']==y['strict'] and x['raw_result']['history']==y['raw_result']['history'] and x['raw_result']['execution_history']==y['raw_result']['execution_history'] and x['raw_result']['proposal_trace']==y['raw_result']['proposal_trace'] and x['raw_result']['token_usage']==y['raw_result']['token_usage']
 source_logging_errors=len(log.records);log.records=[];rs=replay('mixed49-plan-TRAIN-s49',p);out=R/'mixed49-plan-TRAIN-s49';assert not (out/'traces.jsonl').exists() and not (out/'metrics.json').exists()
 (out/'traces.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False,default=str)+'\n' for x in rs));write(out/'metrics.json',[x['metrics'] for x in rs]);counts={}
 for world in ['trainA','trainB']:
  ts=[x for x in rs if x['world']==world];counts[world]={'runs':48,'primary_strict':sum(x['strict'] for x in ts),'execution_strict':sum(x['execution_strict'] for x in ts),'FF':sum(x['metrics']['false_finish'] for x in ts),'parse_valid':sum(x['raw_result']['plan_parse_valid'] for x in ts),'tool_exceptions':sum(len(x['raw_result']['tool_dispatch_errors']) for x in ts),'total_tokens':sum(x['raw_result']['token_usage']['total_tokens'] for x in ts)}
 hashes={str(f):sha(f) for f in [R/'mixed49-plan-TRAIN-s49-job.json',R/'mixed49-plan-TRAIN-s49.log',out/'predictions.jsonl',out/'before-execution.json',out/'traces.jsonl',out/'metrics.json']}
 write(S/'CPU-recovery-receipt.json',{'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'label':'mixed49-plan-TRAIN-s49','original_GPU_exit_code':1,'original_GPU_predictions_complete':48,'CPU_replayed_full_trials':96,'source96_history_semantic_metrics_prediction_usage_identical':True,'source_parity_excludes_new_CPU_tool_and_framework_timing':['total_time_ms','total_tool_ms','framework_overhead_ms','avg_tool_ms'],'source_logger_error_observations':source_logging_errors,'mixed_logger_error_observations':len(log.records),'literal_error_messages':log.records,'new_model_predictions':0,'new_learning_steps':0,'counts':counts,'sha256':hashes,'original_failure_and_predictions_preserved':True,'tool_errors_and_skipped_proposals_remain_failures':True,'replayed_tool_timing_is_new_CPU_timing_original_model_latency_retained':True});print(counts)
if __name__=='__main__':main()
