"""Independent closed-study audit plus termination/cost/rejection accounting. CPU only."""
from pathlib import Path
import os,sys,json,datetime,collections,hashlib,tarfile
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['STORE_BACKEND']='memory';os.environ['FIGURE_ANALYSIS_BACKEND']='extractive';sys.path[:0]=['AgenticArxiv','scripts']
from tools.bootstrap import require_all_tools
require_all_tools('R032 full evidence closure')
from round031_study_core import R,read,write,sha,collect,compare,success_checks
N=R.parent/'round-032-2026-10-10'

def main():
 N.mkdir(exist_ok=True);assert not (N/'closure-independent-audit.json').exists();s=collect(full=True);o=read(R/'study-outcome.json');p=read(R/'experiment-plan.json');assert s['counts']==o['counts'] and s['strict_per_task']==o['strict_per_task'] and s['evidence_sha256']==o['evidence_sha256'];assert success_checks(s['counts'])==o['rule_checks'] and not o['registered_rule_met'] and o['fixed_recipe']=='closed';assert len(o['comparisons'])==48
 comparisons={}
 for f in o['comparisons']:
  v=read(f);assert v==compare(s,v['seed'],v['group'],v['before'],v['after']);comparisons[f]=sha(f)
 jobs=[read(R/(g+'-'+m+'-s'+str(seed)+'-job.json')) for seed in p['generation_seeds'] for g in p['groups'] for m in p['modes']];assert len(jobs)==48 and all(j['exit_code']==0 and j['gpu_index'] in [4,7] for j in jobs)
 aggregate={};diagnostics={};cases=[];allproposals=0
 for seed in p['generation_seeds']:
  diagnostics[str(seed)]={}
  for group in p['groups']:
   diagnostics[str(seed)][group]={}
   for mode in p['modes']:
    c=s['counts'][str(seed)][group][mode];d={'runs':0,'proposal_terminations':collections.Counter(),'execution_terminations':collections.Counter(),'vetoed_action_names':collections.Counter(),'contract_reasons':collections.Counter(),'vetoed_runs':0,'rejected_FINISH_runs':0,'forced_stop_with_unfinished_contract':0,'successful_runs_with_veto':0,'full_proposal_vs_execution_strict_difference':0,'real_observation_commit_failures':0,'repeated_denied_FINISH_runs':0,'prompt_tokens':0,'completion_tokens':0,'reported_total_tokens':0,'summed_response_total_tokens':0,'LLM_calls':0,'vetoes':0,'actual_tool_attempts':0,'supported_runs':0,'execution_strict':c['execution_strict'],'execution_FF':c['execution_FF'],'proposal_strict':c['proposal_strict'],'proposal_FF':c['proposal_FF']}
    traces=[json.loads(x) for x in (R/(group+'-'+mode+'-s'+str(seed))/'traces.jsonl').read_text().splitlines()];extra={(x['task_id'],x['trial']):x for x in read(R/(group+'-'+mode+'-s'+str(seed))/'execution-summary.json')['details']}
    for x in traces:
     raw=x['raw_result'];z=extra[x['task_id'],x['trial']];calls=raw['proposal_trace'];veto=[y['event'] for y in calls if y['event'].get('vetoed')];finish=[e for e in veto if e['action']=='FINISH'];pt=raw['history'][-1]['action'];et=raw['execution_history'][-1]['action'];d['runs']+=1;d['proposal_terminations'][pt]+=1;d['execution_terminations'][et]+=1;d['vetoed_runs']+=bool(veto);d['rejected_FINISH_runs']+=bool(finish);d['repeated_denied_FINISH_runs']+=len(finish)>=2;d['successful_runs_with_veto']+=bool(veto) and z['execution_strict'];d['full_proposal_vs_execution_strict_difference']+=z['proposal_strict']!=z['execution_strict'];d['real_observation_commit_failures']+=sum(y['event'].get('observation_commit') is False for y in calls);d['forced_stop_with_unfinished_contract']+=et=='FORCE_STOP' and raw['contract_supported'] and bool(calls[-1]['event'].get('state_after',calls[-1]['event'].get('state_before',{})).get('completed_steps',0)<calls[-1]['event'].get('state_after',calls[-1]['event'].get('state_before',{})).get('requested_steps',0))
     for e in veto:
      a=e['action'];d['vetoed_action_names'][a if isinstance(a,str) else a.get('name','unknown')]+=1;d['contract_reasons'][e['contract_reason']]+=1
     usage=[y['response']['usage'] for y in calls];pu=sum(y['prompt_tokens'] for y in usage);cu=sum(y['completion_tokens'] for y in usage);total=sum(y['total_tokens'] for y in usage);assert pu+cu==total==raw['token_usage']['total_tokens'];assert pu==raw['token_usage']['prompt_tokens'] and cu==raw['token_usage']['completion_tokens'];d['prompt_tokens']+=pu;d['completion_tokens']+=cu;d['reported_total_tokens']+=total;d['summed_response_total_tokens']+=total
     for k in ['LLM_calls','vetoes','actual_tool_attempts']:d[k]+=z[k]
     d['supported_runs']+=raw['contract_supported'];allproposals+=len(calls)
     if finish and len(cases)<3 and group=='newdev' and seed==45:
      cases.append({'diagnostic_only_not_training':True,'task_id':x['task_id'],'trial':x['trial'],'seed':seed,'group':group,'mode':mode,'termination':et,'repeated_denied_FINISH':len(finish),'proposal_strict':z['proposal_strict'],'execution_strict':z['execution_strict'],'token_usage':raw['token_usage'],'events':[{'action':e['action'],'reason':e['contract_reason'],'state_before':e.get('state_before')} for e in finish],'source_trace_sha256':sha(R/(group+'-'+mode+'-s'+str(seed))/'traces.jsonl')})
    assert d['reported_total_tokens']==c['total_tokens'] and d['LLM_calls']==c['LLM_calls'] and d['vetoes']==c['vetoes'];assert d['full_proposal_vs_execution_strict_difference']==0;diagnostics[str(seed)][group][mode]=d
    agg=aggregate.setdefault(mode,collections.Counter())
    for k in ['runs','vetoed_runs','rejected_FINISH_runs','repeated_denied_FINISH_runs','forced_stop_with_unfinished_contract','successful_runs_with_veto','full_proposal_vs_execution_strict_difference','prompt_tokens','completion_tokens','reported_total_tokens','LLM_calls','vetoes','actual_tool_attempts','supported_runs','execution_strict','execution_FF','proposal_strict','proposal_FF']:agg[k]+=d[k]
    for k,v in d['execution_terminations'].items():agg['terminal_'+k]+=v
 assert allproposals==sum(v['LLM_calls'] for v in aggregate.values());assert aggregate['guard']['successful_runs_with_veto']==0 and aggregate['guard']['full_proposal_vs_execution_strict_difference']==0
 evidence={**s['evidence_sha256'],**comparisons,str(R/'study-outcome.json'):sha(R/'study-outcome.json'),str(R/'experiment-plan.json'):sha(R/'experiment-plan.json'),str(R/'CPU-completion-amendment.json'):sha(R/'CPU-completion-amendment.json')}
 for f,h in read(R/'CPU-completion-amendment.json')['sha256'].items():assert sha(f)==h
 result=dict(s,checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,rule_checks=o['rule_checks'],registered_rule_met=False,fixed_recipe_closed=True,comparisons_verified=48,all_jobs_exit0_authorized48=True,total_model_proposals=allproposals,all_response_token_usage_resummed=True,diagnostics=diagnostics,aggregate=aggregate,diagnostic_examples=cases,evidence_sha256=evidence,new_learning_steps=0,new_model_predictions=0,no_tuning_after_results=True,source_sft_remains_default=True)
 write(N/'closure-independent-audit.json',result);print(json.dumps({'passed':True,'raw_trials':2112,'comparisons':48,'full_model_proposals':allproposals,'rule_checks':o['rule_checks'],'aggregate':aggregate},ensure_ascii=False))
if __name__=='__main__':main()
