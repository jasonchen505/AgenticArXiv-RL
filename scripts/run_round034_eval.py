import os,sys,json,argparse
from pathlib import Path
sys.path[:0]=['AgenticArxiv','scripts']
from round034_core import R,read,rows,write,verify,sha
from run_round032_parity import Runner
from round033_plan_agent import PlanAgent
from tools.bootstrap import require_all_tools
from agents.prompt_templates import build_visible_setup_context
from benchmark.semantic_oracle import attach_expected_paper_ids
from benchmark.metrics import extract_metrics,is_strict_success
def runner(p,s,world):return Runner(mode='native',agent_types=['regex'],repeat=s['repeat'],model=p['model_paths'][s['model']],session_prefix='bench_r20261003_075529',offline=True,snapshot=p['worlds'][world],llm_backend='transformers',local_device='cuda',local_dtype='bfloat16',generation_seed=s['seed'],prompt_policy='baseline')
def main():
 a=argparse.ArgumentParser();a.add_argument('--label',required=True);label=a.parse_args().label;p=read(R/'experiment-plan.json');verify(p);freeze=read(R/'all-finals-before-inference.json')
 for f,h in freeze['sha256'].items():assert sha(f)==h,f
 s=p['eval_jobs'][label];require_all_tools('R034 mixed supervision fixed comparison');tasks=read(s['tasks']);out=R/label;assert not out.exists();rs=[]
 if s['route']=='native':
  r=runner(p,s,s['world']);results=r.run_all(tasks)
  for x in results:
   assert not x.error;rs.append({'task_id':x.task_id,'world':s['world'],'trial':x.trial,'session_id':x.session_id,'raw_result':x.raw_result,'metrics':x.metrics.to_dict(),'strict':is_strict_success(x.metrics)})
 else:
  r=runner(p,s,'trainA');predictions=[]
  for t in tasks:
   for trial in range(s['repeat']):
    r._start_generation_stream(t['id'],trial);a=PlanAgent(r.llm_client,side_effect_mgr=r._side_effects(),env=r._tool_env());q,ms=a.generate(t['task'],p['model_paths'][s['model']],build_visible_setup_context(t));predictions.append({'task_id':t['id'],'trial':trial,'prediction':q,'llm_ms':ms});print('Plan prediction',len(predictions),'/',len(tasks)*s['repeat'],t['id'],trial,flush=True)
  out.mkdir();f=out/'predictions.jsonl';f.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in predictions));write(out/'before-execution.json',{'all_predictions_frozen_before_either_world':True,'predictions':len(predictions),'sha256':sha(f)});byid={(x['task_id'],x['trial']):x for x in predictions}
  for world in ['trainA','trainB']:
   r=runner(p,s,world)
   for t in attach_expected_paper_ids(tasks,p['worlds'][world]):
    for trial in range(s['repeat']):
     r._reset_offline_trial_state();session='bench_r20261003_075529_'+t['id']+'_regex_'+str(trial);a=PlanAgent(None,side_effect_mgr=r._side_effects(),env=r._tool_env());q=byid[t['id'],trial];z=a.execute_prediction(t['task'],q['prediction'],q['llm_ms'],session);m=extract_metrics(t,z,'single_plan',trial,session);em=extract_metrics(t,dict(z,history=z['execution_history']),'executed_single_plan',trial,session);rs.append({'task_id':t['id'],'world':world,'trial':trial,'session_id':session,'raw_result':z,'metrics':m.to_dict(),'execution_metrics':em.to_dict(),'strict':is_strict_success(m),'execution_strict':is_strict_success(em),'one_prediction_reused_two_worlds':True,'per_logical_run_full_prediction_cost':True})
 if not out.exists():out.mkdir()
 assert len(rs)==s['expected_full_trials'];(out/'traces.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False,default=str)+'\n' for x in rs));write(out/'metrics.json',[x['metrics'] for x in rs]);print('Completed',label,len(rs),flush=True)
if __name__=='__main__':main()
