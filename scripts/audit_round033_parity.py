"""Independent, CPU-only closure of the completed R032 input sanity run."""
import os,sys,json,datetime,collections
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',STORE_BACKEND='memory',FIGURE_ANALYSIS_BACKEND='extractive')
sys.path[:0]=['AgenticArxiv','scripts']
from round031_study_core import read,write,sha
from run_round032_parity import N,verify
from arm_blind_observations import ArtifactAddressMap
from tools.bootstrap import require_all_tools
from benchmark.metrics import extract_metrics,is_strict_success
from benchmark.semantic_oracle import attach_expected_paper_ids
R=N.parent/'round-033-2026-10-10'
def main():
 R.mkdir(exist_ok=True);assert not (R/'parity-independent-audit.json').exists();require_all_tools('R033 independent R032 closure');p=read(N/'parity-plan.json');verify(p);o=read(N/'parity-outcome.json');assert o['passed_input_parity_gate'];tasks={x['id']:x for x in attach_expected_paper_ids(read(p['tasks']),p['snapshot'])};traces={};counts={};hashes={};steps=0
 for mode in p['modes']:
  job=read(N/('parity-'+mode+'-job.json'));assert job['exit_code']==0 and job['gpu_index'] in [4,7] and job['started_at_utc']>p['registered_at_utc'];out=N/('parity-'+mode);raw=[json.loads(x) for x in (out/'traces.jsonl').read_text().splitlines()];assert len(raw)==64 and {x['task_id'] for x in raw}==set(tasks);traces[mode]={x['task_id']:x for x in raw};summary=read(out/'summary.json');details={x['task_id']:x for x in summary['details']};root=Path('data/repro_round032')/mode;mapper=ArtifactAddressMap({n:str(root/n) for n in ['raw','translated','figures']});c=collections.Counter()
  for x in raw:
   z=x['raw_result'];assert z['history']==x['history']==z['execution_history'];calls=z['proposal_trace'];assert len(calls)==z['LLM_calls']<=z['max_LLM_calls'];assert z['guard_vetoes']==0
   usage=collections.Counter()
   for q in calls:
    assert q['request']['temperature']==.1 and q['request']['max_tokens']==1000;assert 'data/repro_round032' not in json.dumps(q['request'],ensure_ascii=False)
    usage.update(q['response']['usage'])
   assert dict(usage)==z['token_usage'] and usage['prompt_tokens']+usage['completion_tokens']==usage['total_tokens'];m=extract_metrics(tasks[x['task_id']],z,'regex',x['trial'],x['session_id']);d=details[x['task_id']]
   for attr,key in [('task_completed','completed'),('termination_type','termination'),('tool_call_accurate','tool_accurate'),('parse_failures','parse_fail'),('tool_exec_failures','tool_fail'),('false_finish','false_finish'),('arg_score','arg_score'),('ref_score','ref_score'),('total_tokens','tokens')]:assert getattr(m,attr)==d[key]
   c.update(runs=1,strict=int(is_strict_success(m)),false_finish=int(m.false_finish),LLM_calls=len(calls),**dict(usage))
   for y in z['artifact_observations']:assert mapper.public(y['actual_result'])==y['public_result'];steps+=1
  counts[mode]=dict(c)
  for f in [N/('parity-'+mode+'-job.json'),out/'traces.jsonl',out/'summary.json']:hashes[str(f)]=sha(f)
 a,b=[traces[m] for m in p['modes']];pairs=[]
 for tid,x in a.items():
  u=x['raw_result'];v=b[tid]['raw_result'];assert len(u['proposal_trace'])==len(v['proposal_trace']);assert u['history']==v['history'];assert u['token_usage']==v['token_usage']
  for q,r in zip(u['proposal_trace'],v['proposal_trace']):assert q['request']==r['request'] and q['response']['choices']==r['response']['choices'] and q['response']['usage']==r['response']['usage']
  pairs.append(tid)
 assert o['classification']=={'identical_shared_prefix':64} and len(o['pairs'])==64;assert all(x['native_calls']==x['budget_calls'] for x in o['pairs']);assert counts['native']==counts['budget'];assert counts['native']['strict']==61 and counts['native']['false_finish']==3
 write(R/'parity-independent-audit.json',{'passed':True,'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'full_trials_rescored':128,'full_request_response_choice_history_usage_identical_pairs':64,'raw_public_results_recomputed':steps,'counts':counts,'scoring_drift':0,'frozen_input_hashes_verified':len(p['sha256']),'source_SFT_remains_default':True,'TRAIN_sanity_only_not_Agent_gain':True,'new_model_predictions':0,'new_learning_steps':0,'evidence_sha256':{**hashes,str(N/'parity-plan.json'):sha(N/'parity-plan.json'),str(N/'parity-outcome.json'):sha(N/'parity-outcome.json')}});print(counts)
if __name__=='__main__':main()
