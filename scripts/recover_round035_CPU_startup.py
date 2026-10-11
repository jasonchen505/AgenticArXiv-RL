import os,sys,json,datetime,subprocess
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,'scripts')
from round034_core import R,PY,read,rows,sha,write,verify
S=R.parent/'round-035-2026-10-10'
def main():
 verify(read(S/'CPU-startup-amendment.json'));assert not (S/'CPU-recovery-receipt.json').exists()
 for label in ['source-plan-TRAIN-s49','mixed49-plan-TRAIN-s49']:subprocess.run([PY,'scripts/replay_round035_CPU_worker.py','--label',label],check=True)
 sx=read(S/'source-plan-TRAIN-s49-CPU-replay.json');source=sx['trials'];original=rows(R/'source-plan-TRAIN-s49/traces.jsonl');assert len(source)==len(original)==96
 excluded=['total_time_ms','total_tool_ms','framework_overhead_ms','avg_tool_ms'];untimed=lambda z:{k:v for k,v in z.items() if k not in excluded}
 for x,y in zip(source,original):
  assert untimed(x['metrics'])==untimed(y['metrics']) and untimed(x['execution_metrics'])==untimed(y['execution_metrics']) and x['strict']==y['strict']
  for k in ['history','execution_history','proposal_trace','token_usage']:assert x['raw_result'][k]==y['raw_result'][k],k
 mx=read(S/'mixed49-plan-TRAIN-s49-CPU-replay.json');rs=mx['trials'];out=R/'mixed49-plan-TRAIN-s49';assert len(rs)==96 and not (out/'traces.jsonl').exists() and not (out/'metrics.json').exists()
 (out/'traces.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False,default=str)+'\n' for x in rs));write(out/'metrics.json',[x['metrics'] for x in rs]);counts={}
 for world in ['trainA','trainB']:
  ts=[x for x in rs if x['world']==world];counts[world]={'runs':48,'primary_strict':sum(x['strict'] for x in ts),'execution_strict':sum(x['execution_strict'] for x in ts),'FF':sum(x['metrics']['false_finish'] for x in ts),'parse_valid':sum(x['raw_result']['plan_parse_valid'] for x in ts),'tool_exceptions':sum(len(x['raw_result']['tool_dispatch_errors']) for x in ts),'total_tokens':sum(x['raw_result']['token_usage']['total_tokens'] for x in ts)}
 files=[R/'mixed49-plan-TRAIN-s49-job.json',R/'mixed49-plan-TRAIN-s49.log',out/'predictions.jsonl',out/'before-execution.json',out/'traces.jsonl',out/'metrics.json',S/'CPU-startup-amendment.json']+[S/(l+'-CPU-replay.json') for l in ['source-plan-TRAIN-s49','mixed49-plan-TRAIN-s49']]+[Path('scripts')/n for n in ['replay_round035_CPU_worker.py','recover_round035_CPU_startup.py']]
 write(S/'CPU-recovery-receipt.json',{'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'label':'mixed49-plan-TRAIN-s49','original_GPU_exit_code':1,'original_GPU_predictions_complete':48,'CPU_replayed_full_trials':96,'source96_history_semantic_metrics_prediction_usage_identical':True,'source_parity_excludes_new_CPU_tool_and_framework_timing':excluded,'source_logger_error_observations':len(sx['literal_error_messages']),'mixed_logger_error_observations':len(mx['literal_error_messages']),'literal_error_messages':mx['literal_error_messages'],'new_model_predictions':0,'new_learning_steps':0,'counts':counts,'sha256':{str(f):sha(f) for f in files},'original_failure_and_predictions_preserved':True,'tool_errors_and_skipped_proposals_remain_failures':True,'CPU_context_fixture_matches_original_GPU_startup_config':True,'replayed_tool_timing_is_new_CPU_timing_original_model_latency_retained':True})
 print(counts)
if __name__=='__main__':main()
