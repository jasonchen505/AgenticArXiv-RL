from pathlib import Path
import json,hashlib,datetime,collections,sys
R=Path('artifacts/reproduction/round-031-2026-10-09');PY='.venv-repro/bin/python'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def verify(p):
 for f,h in {**p['sha256'],**p['source_model_freeze']['sha256']}.items():assert sha(f)==h,f

def collect(full=False):
 sys.path[:0]=['AgenticArxiv','scripts'];from benchmark.metrics import extract_metrics,is_strict_success
 from benchmark.semantic_oracle import attach_expected_paper_ids
 p=read(R/'experiment-plan.json');verify(p);counts={};per_task={};hashes={};runs=0;jobs=0
 for seed in p['generation_seeds']:
  counts[str(seed)]={};per_task[str(seed)]={}
  for group,g in p['groups'].items():
   sf,key=g['split'].rsplit(':',1);ids=read(sf)['split'][key];alltasks={t['id']:t for t in attach_expected_paper_ids(read(g['tasks']),g['snapshot'])};tasks={k:alltasks[k] for k in ids};counts[str(seed)][group]={};per_task[str(seed)][group]={}
   for mode in p['modes']:
    label=group+'-'+mode+'-s'+str(seed);out=R/label;jp=R/(label+'-job.json')
    if not jp.exists() or read(jp).get('exit_code')!=0:
     assert not full,'Unclosed '+label;continue
    job=read(jp);assert job['gpu_index'] in [4,7] and job['started_at_utc']>p['registered_at_utc'];summary=read(out/'summary.json');details={(x['task_id'],x['trial']):x for x in summary['details']};extra={(x['task_id'],x['trial']):x for x in read(out/'execution-summary.json')['details']};traces=[json.loads(x) for x in (out/'traces.jsonl').read_text().splitlines()];assert len(details)==len(extra)==len(traces)==len(tasks)*3 and set(details)==set(extra)=={(tid,i) for tid in tasks for i in range(3)}
    seen=set();c={'runs':len(traces),'proposal_strict':0,'proposal_FF':0,'execution_strict':0,'execution_FF':0,'total_tokens':0,'LLM_calls':0,'vetoes':0,'actual_tool_attempts':0,'supported_runs':0,'total_time_ms':0};pt={tid:{'proposal':0,'execution':0} for tid in tasks}
    for x in traces:
     k=(x['task_id'],x['trial']);assert k not in seen;seen.add(k);raw=x['raw_result'];assert x['history']==raw['history'] and raw['mode']==mode;calls=raw['proposal_trace'];assert len(calls)==raw['LLM_calls']<=raw['max_LLM_calls'];assert raw['max_LLM_calls']==(tasks[k[0]].get('max_iterations') or 5)+(0 if mode=='native' else 4);vetoes=sum(y['event'].get('vetoed',False) for y in calls);assert vetoes==raw['guard_vetoes'] and (mode=='guard' or vetoes==0);assert sum('执行校验拒绝（未执行）' in h.get('observation','') for h in raw['history'])==vetoes
     if mode in ['native','budget'] or not raw['contract_supported']:assert raw['execution_history']==raw['history']
     for y in calls:
      assert y['request']['max_tokens']==1000 and y['request']['temperature']==.1 and 'response' in y
     a=extract_metrics(tasks[k[0]],raw,'regex',k[1],x['session_id']);b=extract_metrics(tasks[k[0]],dict(raw,history=raw['execution_history']),'executed_regex',k[1],x['session_id']);d=details[k];z=extra[k]
     mapping={'task_completed':'completed','termination_type':'termination','tool_call_accurate':'tool_accurate','parse_failures':'parse_fail','tool_exec_failures':'tool_fail','false_finish':'false_finish','arg_score':'arg_score','ref_score':'ref_score','terminal_semantics_accurate':'terminal_semantics_accurate','total_tokens':'tokens','total_time_ms':'total_ms'}
     for attr,name in mapping.items():assert getattr(a,attr)==d[name],(label,k,name)
     assert b.to_dict()==z['execution_metrics'];assert is_strict_success(a)==z['proposal_strict'] and a.false_finish==z['proposal_FF'] and is_strict_success(b)==z['execution_strict'] and b.false_finish==z['execution_FF'];assert z['total_tokens']==raw['token_usage']['total_tokens']==a.total_tokens and z['vetoes']==vetoes
     for channel,m in [('proposal',a),('execution',b)]:c[channel+'_strict']+=int(is_strict_success(m));c[channel+'_FF']+=int(m.false_finish);pt[k[0]][channel]+=int(is_strict_success(m))
     for key in ['total_tokens','LLM_calls','vetoes','actual_tool_attempts']:c[key]+=z[key]
     c['supported_runs']+=int(raw['contract_supported']);c['total_time_ms']+=a.total_time_ms
    for channel in ['proposal','execution']:c[channel+'_all3_success_tasks']=sum(t[channel]==3 for t in pt.values())
    counts[str(seed)][group][mode]=c;per_task[str(seed)][group][mode]=pt;runs+=c['runs'];jobs+=1
    for f in [jp]+[out/n for n in ['summary.json','execution-summary.json','execution_protocol.json','traces.jsonl']]:hashes[str(f)]=sha(f)
 if full:assert jobs==p['inference_jobs']==48 and runs==p['total_trial_scores']==2112
 return {'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'counts':counts,'strict_per_task':per_task,'closed_jobs':jobs,'independently_rescored':runs,'proposal_and_execution_drift':0,'scientific_inputs_verified':len(p['sha256']),'source_model_files_verified':len(p['source_model_freeze']['sha256']),'learning_steps':0,'evidence_sha256':hashes}

def success_checks(c):
 rules={'fresh_execution_gain3_vs_native_budget':True,'fresh_execution_no_worse_state':True,'all_proposal_strict_no_worse_budget':True,'olddev_proposal_no_worse_native':True,'all_execution_FF_no_worse_budget_state':True,'fresh_total_tokens_at_most1_5x_budget':True}
 for seed,groups in c.items():
  for group,m in groups.items():
   n,b,s,g=[m[k] for k in ['native','budget','state','guard']]
   rules['all_proposal_strict_no_worse_budget'] &= g['proposal_strict']>=b['proposal_strict']
   rules['all_execution_FF_no_worse_budget_state'] &= g['execution_FF']<=min(b['execution_FF'],s['execution_FF'])
   if group=='olddev':rules['olddev_proposal_no_worse_native'] &= g['proposal_strict']>=n['proposal_strict']
   if group in ['freshA','freshB']:
    rules['fresh_execution_gain3_vs_native_budget'] &= g['execution_strict']>=max(n['execution_strict'],b['execution_strict'])+3
    rules['fresh_execution_no_worse_state'] &= g['execution_strict']>=s['execution_strict']
    rules['fresh_total_tokens_at_most1_5x_budget'] &= 2*g['total_tokens']<=3*b['total_tokens']
 return rules

def compare(s,seed,group,before,after):
 m=s['counts'][str(seed)][group];pt=s['strict_per_task'][str(seed)][group];assert set(pt[before])==set(pt[after]);channels={}
 for channel in ['proposal','execution']:
  channels[channel]={'strict_change':m[after][channel+'_strict']-m[before][channel+'_strict'],'FF_change':m[after][channel+'_FF']-m[before][channel+'_FF'],'task_gains':sum(pt[before][k][channel]<3 and pt[after][k][channel]==3 for k in pt[before]),'task_losses':sum(pt[before][k][channel]==3 and pt[after][k][channel]<3 for k in pt[before])}
 return {'seed':seed,'group':group,'before':before,'after':after,'models':{before:m[before],after:m[after]},'channels':channels,'same_task_trial_stream_seed':True,'execution_is_secondary_architecture_metric':True,'no_learning_or_significance_claim':True}
