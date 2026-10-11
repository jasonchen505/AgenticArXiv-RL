#!/usr/bin/env python3
"""TRAIN-only native GRPO lr0 eligibility sampling; no automatic learning after the gate."""
from pathlib import Path
import collections,copy,datetime,fcntl,hashlib,json,os,signal,statistics,subprocess,sys,tempfile,time
R=Path('artifacts/reproduction/round-028-2026-10-09');OLD=R.parent/'round-024-2026-10-07';CLOSED=R.parent/'round-027-2026-10-08';OUT=Path('outputs/repro_round028/program_probe');PY='.venv-repro/bin/python'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def verify(plan):
 for p,h in plan['sha256'].items():assert sha(p)==h,p
def state(phase,**kw):write(R/'probe-status.json',dict(phase=phase,updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),learning_updates=0,**kw))
def admission(gpu):
 assert gpu in [4,7];uuid,name,free=[x.strip() for x in subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-gpu=uuid,name,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')];raw=subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True);active=sum(x.split(',')[0].strip()==uuid for x in raw.splitlines());return name=='NVIDIA GeForce RTX 4090' and int(free)>=23000 and active==0,{'free_mib':int(free),'compute_processes':active}
def register():
 os.environ['CUDA_VISIBLE_DEVICES']='';assert not (R/'program-probe-plan.json').exists() and not OUT.exists();closure=read(R/'closure-independent-audit.json');assert closure['passed'] and closure['fixed_recipe_closed']
 tasks=read(OLD/'train-tasks.json');oracles=read(OLD/'train-oracles.json');assert len(tasks)==len(oracles)==64 and len({t['provenance']['family'] for t in tasks})==8
 train_ids={t['id'] for t in tasks};assert not train_ids&{t['id'] for t in read(OLD/'all-tasks.json') if t['provenance']['family'] in ['CSF','SCF','CSFA','SCFA']}
 for label in ['confirmA','confirmB']:assert not train_ids&{t['id'] for t in read(R.parent/'round-026-2026-10-08'/(label+'-oracles.json'))}
 snapshot=Path('data/repro_round024/trainA-snapshot.json');paths=[Path(__file__),OLD/'train-tasks.json',OLD/'train-oracles.json',snapshot,R/'closure-independent-audit.json']+[Path('AgenticArxiv')/p for p in ['rl/train_grpo.py','rl/multiturn_env.py','rl/rollout_audit.py','rl/grpo_reward.py','rl/reward.py','rl/trl_compat.py','agents/prompt_templates.py','benchmark/metrics.py','benchmark/semantic_oracle.py']]+[Path('scripts/run_round020_gpu_job.py')]
 write(R/'program-probe-preparation.json',{'registered_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'purpose':'Native source-SFT online reward eligibility on all64 frozen TRAIN compositions, not another CE/KL coefficient or newheldoutcase repair.','CPU_gold_first':True,'families':sorted({t['provenance']['family'] for t in tasks}),'sha256':{str(p):sha(p) for p in paths},'model_predictions':0,'learning_updates':0})
 os.environ.update(STORE_BACKEND='memory',FIGURE_ANALYSIS_BACKEND='extractive');sys.path[:0]=['AgenticArxiv','scripts'];gold=[]
 with tempfile.TemporaryDirectory(prefix='program028-gold-') as tmp:
  for key in ['PDF_RAW_PATH','PDF_TRANSLATED_PATH','PDF_FIGURES_PATH']:os.environ[key]=str(Path(tmp)/key.lower())
  from tools.bootstrap import require_all_tools
  from rl.multiturn_env import AgenticArxivMultiTurnEnv
  from rl.reward import RewardCalculator
  from benchmark.semantic_oracle import _call
  from benchmark.task_spec import reference_terminal_thought
  from benchmark.metrics import is_strict_success
  require_all_tools('Round028 native TRAIN program gold');calc=RewardCalculator(curriculum_steps=0)
  for t in oracles:
   env=AgenticArxivMultiTurnEnv(snapshot);env.reset(task_id=t['id']);history=[]
   for name,args in zip(t['expected_tools'],t['expected_tool_args']):
    result=_call(env,name,args);history.append({'thought':'执行训练契约标准步骤','action':json.dumps({'name':name,'args':args},ensure_ascii=False),'observation':json.dumps(result,ensure_ascii=False,default=str)})
   history.append({'thought':reference_terminal_thought(t),'action':'FINISH','observation':'任务完成'});b,m=calc.compute_reward_breakdown(t,{'history':history});assert is_strict_success(m) and b.total>=.99
   gold.append({'task_id':t['id'],'strict':True,'legacy_reward':b.total})
 write(R/'program-probe-native-gold.json',{'CPU_only':True,'gold_passed':64,'families':8,'model_predictions':0,'results':gold})
 plan=read(CLOSED/'experiment-plan.json');files=plan['sha256'].copy()
 for p,h in files.items():assert sha(p)==h,p
 for p in paths+[R/'program-probe-preparation.json',R/'program-probe-native-gold.json']:files[str(p)]=sha(p)
 for p in Path(plan['source_model']).rglob('*'):
  if p.is_file():files[str(p)]=sha(p)
 probe={'registered_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_model':plan['source_model'],'tasks':str(OLD/'train-tasks.json'),'oracles':str(OLD/'train-oracles.json'),'snapshot':str(snapshot),'all64_existing_TRAIN_compositions_used':True,'families':8,'groups_per_task':1,'generations_per_group':4,'model_rollouts_planned':256,'native_loop_steps':64,'lr':0.,'learning_updates':0,'seed':46,'temperature':1.,'max_turns':6,'max_completion_length':2048,'beta':.04,'native_trainer':'Unmodified native GRPOTrainer / original multiturn rollout and legacy reward path','loss_type':'grpo','scale_rewards':'group','dynamic_sampling':False,'mask_truncated_completions':True,'allow_zero_variance':True,'gold_standard_valid':64,'eligibility_rule':{'all256_samples_and64_single_groups_closed':True,'minimum_mixed_strict_groups':16,'minimum_nonzero_legacy_reward_variance_groups':16,'maximum_token_clipped_samples':4,'all392policy_and_reference_tensors_equal_source':True},'gate_effect':'Eligibility is NOT effectiveness. If any gate fails close this fixedpilot;do not change seed/temp/turns/cases to pass. If allpass, nextstep requires a separate complete matched nativelearning protocol and freshconfirmation preparation. Never automatic train from rawfailures or select onlyvariance-positive tasks.','source_scope':'Original sourceSFT chosen after full1512 confirmed it exceeds allresumedmodels onfreshCFAS/CFSA. All64 tasks/families frozen before this newdiagnostic; noheldouttext/traces/outputfailures build tasks. Prior GRPO originaltrain/hard/broad/reward routes alreadyran and didnot establishstablegain; this tests online eligibility on addedprogram coverage, not proofGRPOwasmissing.','no_candidate_or_heldout_inference':True,'no_new_learning_permitted':True,'sampling_not_benchmark_temp':True,'only_gpu':[4,7],'sha256':files}
 write(R/'program-probe-plan.json',probe);print(json.dumps({k:probe[k] for k in ['registered_at_utc','native_loop_steps','model_rollouts_planned','lr']},indent=2))
def worker():
 plan=read(R/'program-probe-plan.json');verify(plan);gpu=int(os.environ.get('CUDA_VISIBLE_DEVICES','-1'));assert admission(gpu)[0] and plan['lr']==0. and not OUT.exists()
 sys.path.insert(0,'AgenticArxiv');from rl import train_grpo as module
 tasks=read(plan['tasks'])
 def load(task_set,split):
  assert task_set=='expanded' and split==plan['tasks']+':train_only';return copy.deepcopy(tasks)
 module._load_tasks=load
 module.main(model=plan['source_model'],output_dir=str(OUT),epochs=1,max_steps=64,batch_size=4,grad_accum=1,lr=0.,beta=.04,reward_curriculum_steps=0,reward_objective='legacy',num_generations=4,max_completion_length=2048,max_turns=6,temperature=1.,loss_type='grpo',mask_truncated_completions=True,importance_sampling_level='token',scale_rewards='group',dynamic_sampling=False,seed=46,snapshot=plan['snapshot'],task_set='expanded',split=plan['tasks']+':train_only',allow_zero_variance=True,canary_steps=0,verify=False,report_to='none',save_rollout_traces=True)
def finalize():
 os.environ['CUDA_VISIBLE_DEVICES']='';plan=read(R/'program-probe-plan.json');verify(plan);assert read(R/'program-probe-job.json')['exit_code']==0
 from safetensors.torch import load_file
 import torch
 source=load_file(plan['source_model']+'/adapter_model.safetensors',device='cpu');end=load_file(str(OUT/'final/adapter_model.safetensors'),device='cpu');ref=load_file(str(OUT/'final/ref/adapter_model.safetensors'),device='cpu');assert set(source)==set(end)==set(ref) and len(source)==392 and all(torch.equal(source[k],end[k]) and torch.equal(source[k],ref[k]) for k in source)
 m=read(OUT/'final/training_manifest.json');assert m['actual_steps']==64 and m['learning_rate']==0. and m['frozen_policy_probe']
 sys.path.insert(0,'AgenticArxiv');from benchmark.metrics import extract_metrics,is_strict_success
 from rl.reward import RewardCalculator
 calc=RewardCalculator(curriculum_steps=0);tasks={t['id']:t for t in read(plan['oracles'])};raw=[json.loads(x) for x in (OUT/'rollout_traces.jsonl').read_text().splitlines()];groups=collections.defaultdict(list);strict=clipped=0
 assert len(raw)==256 and collections.Counter(x['task_id'] for x in raw)==dict.fromkeys(tasks,4)
 for x in raw:
  task=tasks[x['task_id']];b,metric=calc.compute_reward_breakdown(task,x['trajectory']);assert abs(b.total-x['reward'])<1e-5;ok=is_strict_success(metric);strict+=int(ok);clipped+=int(x['trajectory'].get('clipped',False));groups[x['task_id'],x['training_step'],x['batch_index'],x['group_index']].append((x['generation_index'],int(ok),x['reward']))
 assert len(groups)==64 and all(len(v)==4 and {x[0] for x in v}=={0,1,2,3} for v in groups.values())
 mixed=sum(0<sum(x[1] for x in v)<4 for v in groups.values());variable=sum(statistics.pstdev(x[2] for x in v)>1e-8 for v in groups.values());gate=mixed>=16 and variable>=16 and clipped<=4
 outcome={'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actual_rollouts':256,'native_lr0_loop_steps':64,'learning_updates':0,'source_policy_equal_tensors':392,'reference_equal_source_tensors':392,'strict_successes':strict,'strict_failures':256-strict,'mixed_strict_groups':mixed,'nonzero_legacy_variance_groups':variable,'token_clipped_samples':clipped,'eligibility_gate_met':gate,'fixed_pilot_state':'eligible_prepare_separate_learning_protocol' if gate else 'closed','no_model_improvement_claim':True,'no_automatic_learning':True,'all_samples_independently_rescored':256,'strict_and_reward_drift':0,'raw_traces_sha256':sha(OUT/'rollout_traces.jsonl'),'plan_sha256':sha(R/'program-probe-plan.json')};write(R/'program-probe-outcome.json',outcome);state('completed',**outcome)
def controller():
 os.environ['CUDA_VISIBLE_DEVICES']=''
 with (R/'probe.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);plan=read(R/'program-probe-plan.json');verify(plan);assert not (R/'program-probe-job.json').exists();deadline=time.monotonic()+1800
  while True:
   candidates=[g for g in [7,4] if admission(g)[0]]
   if candidates:gpu=candidates[0];break
   state('waiting_for_capacity');assert time.monotonic()<deadline,'Finite GPU wait expired';time.sleep(30)
  scratch=Path('data/repro_round028/probe_scratch');env=dict(os.environ)
  for key,name in [('PDF_RAW_PATH','raw'),('PDF_TRANSLATED_PATH','translated'),('PDF_FIGURES_PATH','figures')]:env[key]=str((scratch/name).resolve())
  state('native_lr0_sampling',physical_gpu=gpu,planned_samples=256)
  child=subprocess.Popen([PY,'scripts/run_round020_gpu_job.py','--record',str(R/'program-probe-job.json'),'--log',str(R/'program-probe.log'),'--gpu',str(gpu),'--',PY,'-u','scripts/round028_program_probe.py','worker'],env=env,start_new_session=True)
  try:assert child.wait(timeout=5400)==0,'Native probe failed; preserve evidence'
  finally:
   if child.poll() is None:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=15)
  finalize()
if __name__=='__main__':
 mode=sys.argv[1]
 try:globals()[mode]()
 except Exception as e:
  if mode=='controller':state('failed_or_timeout',error=str(e))
  raise
