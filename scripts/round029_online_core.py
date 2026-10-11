#!/usr/bin/env python3
from pathlib import Path
import os,sys,json,hashlib,datetime,collections,re
R=Path('artifacts/reproduction/round-029-2026-10-09');PY='.venv-repro/bin/python';read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def verify(plan):
 for p,h in plan['sha256'].items():assert sha(p)==h,p

def training_audit(plan,full=True):
 os.environ['CUDA_VISIBLE_DEVICES']='';import torch
 from safetensors.torch import load_file
 source=load_file(plan['source_model']+'/adapter_model.safetensors',device='cpu');records={}
 for arm in plan['execution_arms']:
  j=R/(arm+'-train-job.json')
  if not j.exists() or read(j).get('exit_code')!=0:
   assert not full,'Missing completed '+arm;continue
  job=read(j);assert job['gpu_index'] in [4,7];d=Path(plan['model_paths'][arm]);m=read(d/'training_manifest.json');assert m['actual_steps']==64 and m['task_count']==64 and m['seed']==plan['arms'][arm]['seed'] and m['learning_rate']==plan['arms'][arm]['lr'] and m['beta']==.04 and not m['stopped_early'] and not m['dynamic_sampling']['enabled']
  a=load_file(str(d/'adapter_model.safetensors'),device='cpu');ref=load_file(str(d/'ref/adapter_model.safetensors'),device='cpu');assert set(source)==set(a)==set(ref) and len(a)==392 and all(torch.equal(source[k],ref[k]) for k in source);changed=sum(not torch.equal(source[k],a[k]) for k in source);assert changed>0 if plan['arms'][arm]['lr'] else changed==0
  raw=d.parent/'rollout_traces.jsonl';samples=[json.loads(x) for x in raw.read_text().splitlines()];ids={x['id'] for x in read(plan['train_tasks'])};assert len(samples)==256 and collections.Counter(x['task_id'] for x in samples)==dict.fromkeys(ids,4);groups=collections.defaultdict(list)
  for x in samples:groups[x['task_id'],x['training_step'],x['batch_index'],x['group_index']].append(x['generation_index'])
  assert len(groups)==64 and all(sorted(v)==[0,1,2,3] for v in groups.values());records[arm]={'native_optimizer_steps':64,'learning_rate':m['learning_rate'],'learning_update_steps':64 if m['learning_rate'] else 0,'rollouts':256,'policy_changed_tensors':changed,'reference_source_equal_tensors':392,'manifest_sha256':sha(d/'training_manifest.json'),'rollouts_sha256':sha(raw),'job':job}
 return records

def collect(full=False):
 os.environ.update(CUDA_VISIBLE_DEVICES='',STORE_BACKEND='memory');sys.path.insert(0,'AgenticArxiv');from benchmark.metrics import extract_metrics,is_strict_success
 plan=read(R/'experiment-plan.json');verify(plan);freeze=read(R/'all-finals-before-inference.json') if (R/'all-finals-before-inference.json').exists() else None
 if freeze:
  for model in freeze['models'].values():
   for p,h in model['sha256'].items():assert sha(p)==h,p
 counts={};evidence={};per={};total=after=jobs=0
 for group,g in plan['groups'].items():
  tasks={t['id']:t for t in read(g['oracles'])};sf,key=g['split'].rsplit(':',1);wanted=set(read(sf)['split'][key]);counts[group]={};per[group]={}
  for model in plan['models_order']:
   reused=plan['reused_evals'].get(group,{}).get(model);path=Path(reused) if reused else R/(group+'-'+model)
   if not reused:
    jp=R/(group+'-'+model+'-job.json')
    if not jp.exists() or read(jp).get('exit_code')!=0:
     assert not full,'Incomplete '+str(jp);continue
    j=read(jp);assert freeze and j['gpu_index'] in [4,7] and datetime.datetime.fromisoformat(j['started_at_utc'])>datetime.datetime.fromisoformat(freeze['recorded_at_utc']);jobs+=1
   raw=[json.loads(x) for x in (path/'traces.jsonl').read_text().splitlines()];summary=read(path/'summary.json');details={(d['task_id'],d['trial']):d for d in summary['details']};assert len(raw)==len(wanted)*3 and len(details)==len(raw) and not summary['error_count'] and collections.Counter(x['task_id'] for x in raw)==dict.fromkeys(wanted,3);strict=ff=0;task=collections.Counter();seen=set()
   for x in raw:
    k=x['task_id'],x['trial'];assert k not in seen;seen.add(k);d=details[k];m=extract_metrics(tasks[k[0]],{'history':x['history']},'CPU_R029',x['trial'],x['session_id']);good=d['completed'] and d['tool_accurate'] and d['arg_score']==1 and d['ref_score']==1 and d['parse_fail']==0 and d['tool_fail']==0 and d['terminal_semantics_accurate'];assert is_strict_success(m)==good and m.false_finish==d['false_finish'];strict+=int(good);ff+=int(m.false_finish);task[k[0]]+=int(good)
   assert seen==set(details);counts[group][model]={'runs':len(raw),'strict':strict,'false_finish':ff,'all3_success_tasks':sum(x==3 for x in task.values())};per[group][model]=dict(task);total+=len(raw);after+=0 if reused else len(raw)
   for p in [path/'traces.jsonl',path/'summary.json',path/'execution_protocol.json']:evidence[str(p)]=sha(p)
 training=training_audit(plan,full=full);logged={}
 for arm in plan['execution_arms']:
  log=R/(arm+'-train.log')
  if log.exists():logged[arm]=max([int(x) for x in re.findall(r'(\d+)/64',log.read_text(errors='replace'))],default=0)
 if full:assert total==1320 and after==1152 and jobs==26 and len(training)==4
 return {'captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_only':True,'plan_sha256':sha(R/'experiment-plan.json'),'all_scientific_inputs_unchanged':True,'training':training,'logged_native_steps':logged,'actual_completed_learning_steps':sum(x['learning_update_steps'] for x in training.values()),'actual_completed_native_optimizer_steps':sum(x['native_optimizer_steps'] for x in training.values()),'all_new_finals_frozen_before_inference':bool(freeze),'freeze':freeze,'historical_source_runs_reused':168,'independently_rescored':total,'after_baseline_runs_closed':after,'after_baseline_runs_planned':1152,'strict_and_FF_drift':0,'inference_jobs_closed':jobs,'counts':counts,'strict_per_task':per,'evidence_sha256':evidence,'full_evaluation_complete':after==1152,'status':read(R/'experiment-status.json') if (R/'experiment-status.json').exists() else {'phase':'registered_not_started'},'no_candidate_gain_claim_from_partial_scores':after<1152}
def success_checks(c):
 each=lambda f:all(f(str(seed)) for seed in [43,44]);return {'each_seed_each_fresh_world_gain_at_least3_vs_source_and_lr0':each(lambda s:all(c[g]['rl'+s]['strict']>=max(c[g]['sft']['strict'],c[g]['lr0_'+s]['strict'])+3 for g in ['freshA','freshB'])),'oldDev_strict_no_worse_source_and_lr0':each(lambda s:c['olddev']['rl'+s]['strict']>=max(c['olddev']['sft']['strict'],c['olddev']['lr0_'+s]['strict'])),'newDev_strict_gain_at_least3_vs_source_and_no_worse_lr0':each(lambda s:c['newdev']['rl'+s]['strict']>=max(c['newdev']['sft']['strict']+3,c['newdev']['lr0_'+s]['strict'])),'seen_confirmations_strict_no_worse_source_and_lr0':each(lambda s:all(c[g]['rl'+s]['strict']>=max(c[g]['sft']['strict'],c[g]['lr0_'+s]['strict']) for g in ['seenA','seenB'])),'all_six_groups_FF_no_worse_source_and_lr0':each(lambda s:all(c[g]['rl'+s]['false_finish']<=min(c[g]['sft']['false_finish'],c[g]['lr0_'+s]['false_finish']) for g in c))}

def worker_admission(gpu):
 import subprocess
 assert gpu in [4,7];uuid,name,free=[x.strip() for x in subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-gpu=uuid,name,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')];raw=subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True);active=sum(x.split(',')[0].strip()==uuid for x in raw.splitlines());return name=='NVIDIA GeForce RTX 4090' and int(free)>=23000 and active==0
