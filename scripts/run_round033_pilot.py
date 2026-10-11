import os,sys,json,argparse,copy
from pathlib import Path
sys.path[:0]=['AgenticArxiv','scripts']
from round031_study_core import read,write,sha
from round033_plan_agent import PlanAgent
from run_round032_parity import Runner
from benchmark.metrics import extract_metrics,is_strict_success
from benchmark.semantic_oracle import attach_expected_paper_ids
from agents.prompt_templates import build_visible_setup_context
from tools.bootstrap import require_all_tools
R=Path('artifacts/reproduction/round-033-2026-10-10')
def verify(p):
 for f,h in p['sha256'].items():assert sha(f)==h,f
def make_runner(p,world):return Runner(mode='native',agent_types=['regex'],repeat=1,model=p['source_model'],session_prefix='bench_r20261003_075529',offline=True,snapshot=p['worlds'][world],llm_backend='transformers',local_device='cuda',local_dtype='bfloat16',generation_seed=48,prompt_policy='baseline')
def save(out,rows):
 assert not out.exists();out.mkdir();(out/'traces.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False,default=str)+'\n' for x in rows));write(out/'metrics.json',[x['metrics'] for x in rows]);print('Completed',len(rows),'full trials at',out,flush=True)
def main():
 a=argparse.ArgumentParser();a.add_argument('--job',choices=['reactA','reactB','planner'],required=True);job=a.parse_args().job;p=read(R/'pilot-plan.json');verify(p);require_all_tools('R033 fixed source plan pilot');tasks=read(p['tasks']);assert len(tasks)==48;out=R/job;assert not out.exists()
 if job!='planner':
  world='train'+job[-1];runner=make_runner(p,world);rows=[]
  for x in runner.run_all(tasks):
   assert not x.error;z=x.raw_result;rows.append({'task_id':x.task_id,'world':world,'trial':x.trial,'session_id':x.session_id,'raw_result':z,'metrics':x.metrics.to_dict(),'strict':is_strict_success(x.metrics)})
  save(out,rows);return
 runner=make_runner(p,'trainA');predictions=[]
 for i,t in enumerate(tasks):
  runner._start_generation_stream(t['id'],0);agent=PlanAgent(runner.llm_client,side_effect_mgr=runner._side_effects(),env=runner._tool_env());pred,ms=agent.generate(t['task'],p['source_model'],build_visible_setup_context(t));assert 'data/repro_round033' not in json.dumps(pred['request'],ensure_ascii=False);predictions.append({'task_id':t['id'],'prediction':pred,'llm_ms':ms});print('Plan prediction',i+1,'/48',t['id'],flush=True)
 pf=R/'planner-predictions.jsonl';assert not pf.exists();pf.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in predictions));write(R/'planner-predictions-before-execution.json',{'all48_predictions_frozen_before_any_plan_execution':True,'sha256':sha(pf),'real_LLM_calls':48});byid={x['task_id']:x for x in predictions};rows=[]
 for world in p['worlds']:
  runner=make_runner(p,world)
  for t in attach_expected_paper_ids(tasks,p['worlds'][world]):
   runner._reset_offline_trial_state();session='bench_r20261003_075529_'+t['id']+'_regex_0';agent=PlanAgent(None,side_effect_mgr=runner._side_effects(),env=runner._tool_env());q=byid[t['id']];z=agent.execute_prediction(t['task'],q['prediction'],q['llm_ms'],session);m=extract_metrics(t,z,'single_plan',0,session);em=extract_metrics(t,dict(z,history=z['execution_history']),'executed_single_plan',0,session);rows.append({'task_id':t['id'],'world':world,'trial':0,'session_id':session,'raw_result':z,'metrics':m.to_dict(),'execution_metrics':em.to_dict(),'strict':is_strict_success(m),'execution_strict':is_strict_success(em),'one_prediction_reused_in_two_worlds':True,'per_logical_run_cost_full_plan_charged':True})
  print('Executed frozen plans',world,48,flush=True)
 save(out,rows);assert len(rows)==96
if __name__=='__main__':main()
