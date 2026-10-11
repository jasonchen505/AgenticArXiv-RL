import os,sys,json
os.environ['CUDA_VISIBLE_DEVICES']='';sys.path[:0]=['AgenticArxiv','scripts']
from recover_round035_frozen_plans import replay,S
from round034_core import R,read,rows,write
from round035_literal_error_log import install
from tools.bootstrap import require_all_tools
require_all_tools('CPU parity first difference, no LLM generation');install();xs=replay('source-plan-TRAIN-s49',read(R/'experiment-plan.json'));ys=rows(R/'source-plan-TRAIN-s49/traces.jsonl');diff=[]
for x,y in zip(xs,ys):
 row={}
 for kind in ['metrics','execution_metrics']:
  row[kind]={k:{'CPU':v,'original':y[kind].get(k)} for k,v in x[kind].items() if v!=y[kind].get(k)}
 for kind in ['history','execution_history','proposal_trace','token_usage']:
  if x['raw_result'][kind]!=y['raw_result'][kind]:row[kind]={'CPU':x['raw_result'][kind],'original':y['raw_result'][kind]}
 if any(row.get(k) for k in ['history','execution_history','proposal_trace','token_usage']) or any(k not in ['total_time_ms','total_tool_ms','framework_overhead_ms','avg_tool_ms'] for typ in ['metrics','execution_metrics'] for k in row[typ]):diff.append({'task_id':x['task_id'],'world':x['world'],'diff':row})
write(S/'CPU-first-parity-difference.json',{'differing_semantic_or_history_trials':len(diff),'first_difference':diff[:1],'model_predictions':0});print(json.dumps({'different':len(diff),'first':diff[:1]},ensure_ascii=False))
