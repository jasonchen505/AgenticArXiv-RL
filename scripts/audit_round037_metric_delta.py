import os,sys,collections,datetime,json
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',STORE_BACKEND='memory',FIGURE_ANALYSIS_BACKEND='extractive',TOOL_ROUTER='policy');sys.path[:0]=['AgenticArxiv','scripts']
from benchmark.metrics import extract_metrics,is_strict_success
from benchmark.semantic_oracle import attach_expected_paper_ids
from tools.bootstrap import require_all_tools,registered_tool_count
from round036_core import read,rows,write,sha
R=Path('artifacts/reproduction/round-037-2026-10-11')
def main():
 require_all_tools('R037 old traces diagnostic rescore under merged new scorer');changed=[];counts={};total=0
 for roundno,path in [(34,Path('artifacts/reproduction/round-034-2026-10-10')),(36,Path('artifacts/reproduction/round-036-2026-10-11'))]:
  p=read(path/'experiment-plan.json');groups={}
  for label,s in p['eval_jobs'].items():
   worlds=['trainA','trainB'] if s['route']=='plan' else [s['world']];tasks={w:{t['id']:t for t in attach_expected_paper_ids(read(s['tasks']),p['worlds'][w])} for w in worlds};stat=collections.Counter()
   for x in rows(path/label/'traces.jsonl'):
    m=extract_metrics(tasks[x['world']][x['task_id']],x['raw_result'],'single_plan' if s['route']=='plan' else 'regex',x['trial'],x['session_id']);new=is_strict_success(m);old=x['metrics'];stat.update(runs=1,old_strict=int(x['strict']),new_strict=int(new),old_parse=old['parse_failures'],new_parse=m.parse_failures,old_tool_fail=old['tool_exec_failures'],new_tool_fail=m.tool_exec_failures)
    if (new,m.parse_failures,m.tool_exec_failures,m.false_finish)!=(x['strict'],old['parse_failures'],old['tool_exec_failures'],old['false_finish']):changed.append({'round':roundno,'label':label,'task_id':x['task_id'],'world':x['world'],'trial':x['trial'],'old_strict':x['strict'],'new_strict':new,'old_parse':old['parse_failures'],'new_parse':m.parse_failures,'old_tool_fail':old['tool_exec_failures'],'new_tool_fail':m.tool_exec_failures,'old_FF':old['false_finish'],'new_FF':m.false_finish})
    total+=1
   groups[label]=dict(stat)
  counts[str(roundno)]=groups
 write(R/'new-scorer-diagnostic.json',{'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'merged_runtime_registered_tools':registered_tool_count(),'frozen_old_traces_rescored_diagnostically':total,'changed_verdict_rows':len(changed),'changes':changed,'counts':counts,'original_results_not_overwritten':True,'not_a_new_model_evaluation':True,'new_learning_or_model_predictions':0,'no_reserved_tasks_materialized':True,'scorer_SHA256':sha('AgenticArxiv/benchmark/metrics.py')});print({'diagnostic_raw_rows':total,'changed_verdict_rows':len(changed),'registered_tools':registered_tool_count()})
if __name__=='__main__':main()
