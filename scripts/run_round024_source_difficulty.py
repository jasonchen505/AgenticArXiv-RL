#!/usr/bin/env python3
"""One finite frozen SFT diagnostic; no candidate learning or case selection."""
from pathlib import Path
import collections,datetime,fcntl,hashlib,json,os,subprocess,time
R=Path('artifacts/reproduction/round-024-2026-10-07');D=Path('data/repro_round024');PY='.venv-repro/bin/python'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def state(phase,**kw):write(R/'difficulty-status.json',dict(phase=phase,updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),new_training_steps=0,**kw))
def main():
 with (R/'difficulty.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert not (R/'baseline-difficulty-plan.json').exists();audit=read(R/'curriculum-independent-audit.json');assert audit['passed'] and audit['raw_gold_independently_rescored']==112 and audit['train_rows_reconstructed']==352
  tasks=R/'dev-oracles.json';ts=read(tasks);assert len(ts)==16
  split=R/'difficulty-splits.json';write(split,{'split':{'dev':[t['id'] for t in ts]}})
  sources={};prep=read(R/'curriculum-preparation-plan.json');ad=read(R/'curriculum-resource-admission.json')
  sources.update(prep['sha256']);sources.update(prep['asset_sha256']);sources.update(ad['sha256']);sources.update(audit['data_sha256'])
  for p in [split,R/'curriculum-independent-audit.json',R/'curriculum-gold-validation.json',Path(__file__),Path('scripts/run_round020_gpu_job.py'),Path('scripts/run_frozen_task_file.py')]+list(Path('AgenticArxiv').rglob('*.py'))+list(Path('outputs/repro_20261003/sft_e1/final').rglob('*')):
   if p.is_file():sources[str(p)]=sha(p)
  original=read(R.parent/'round-023-2026-10-07/experiment-plan.json')
  for p,h in original['sha256'].items():
   if '/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/' in p:sources[p]=h
  plan={'registered_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'kind':'Fixed sourceSFT development-difficulty gate before any candidate training','tasks':str(tasks),'split':str(split)+':dev','snapshot':str(D/'trainA-snapshot.json'),'source_model':'outputs/repro_20261003/sft_e1/final','source_adapter_sha256':sha('outputs/repro_20261003/sft_e1/final/adapter_model.safetensors'),'repeat':3,'seed':45,'runs':48,'task_families':2,'dtype':'bfloat16','prompt_policy':'baseline','allowed_physical_gpus':[4,7],'inference_admission':{'minimum_free_mib':8192,'shared_allowed':True},'numeric_gate':{'source_dev_strict_at_most_for_future_learning':40,'if_above':'Close fixed curriculum difficulty pilot; do not harden tasks using output failures or start candidate training.','if_at_most':'Enough observed room to preregister a matched795row199step two-seed curriculum-vs-repeat protocol; no automatic training or promotion.','missing_or_error':'No learning; preserve failure.'},'scope':'16tasks are2novelorderedfamilies*8train-derivedparameter combinations on reusedtrainA. Development diagnostic, not16 independent programs, not paper-blind confirmation. No confirmation model outputs this round.','new_training_steps':0,'candidate_models':0,'no_automatic_promotion':True,'sha256':sources}
  write(R/'baseline-difficulty-plan.json',plan)
  def verify():
   for p,h in sources.items():assert sha(p)==h,p
  verify();deadline=time.monotonic()+1800;gpu=None
  while gpu is None:
   for g in [4,7]:
    free=int(subprocess.check_output(['nvidia-smi',f'--id={g}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if free>=8192:gpu=g;break
   if gpu is None:state('waiting_for_inference_capacity');assert time.monotonic()<deadline;time.sleep(30)
  verify();scratch=D/'source_difficulty_scratch';assert not scratch.exists();env=dict(os.environ)
  for k,n in [('PDF_RAW_PATH','raw'),('PDF_TRANSLATED_PATH','translated'),('PDF_FIGURES_PATH','figures')]:env[k]=str((scratch/n).resolve())
  state('source_development_inference',physical_gpu=gpu,planned_runs=48)
  cmd=[PY,'-u','scripts/run_frozen_task_file.py','--tasks',str(tasks),'--tasks_sha256',sha(tasks),'--split',plan['split'],'--snapshot',plan['snapshot'],'--model',plan['source_model'],'--output',str(R/'source-new-dev'),'--prefix','bench_r20261003_075529','--seed','45','--repeat','3']
  subprocess.run([PY,'scripts/run_round020_gpu_job.py','--record',str(R/'source-new-dev-job.json'),'--log',str(R/'source-new-dev.log'),'--gpu',str(gpu),'--']+cmd,timeout=3600,check=True,env=env);verify()
  # CPU import gets explicit no-CUDA; score every completed trial independently.
  os.environ['CUDA_VISIBLE_DEVICES']='';import sys;sys.path.insert(0,'AgenticArxiv')
  from benchmark.metrics import extract_metrics,is_strict_success
  raw=[json.loads(x) for x in (R/'source-new-dev/traces.jsonl').read_text().splitlines()];summary=read(R/'source-new-dev/summary.json');details={(d['task_id'],d['trial']):d for d in summary['details']};byid={t['id']:t for t in ts};assert len(raw)==len(details)==48 and not summary['error_count'] and collections.Counter(x['task_id'] for x in raw)==dict.fromkeys(byid,3)
  strict=ff=0;per=collections.Counter()
  for x in raw:
   t=byid[x['task_id']];m=extract_metrics(t,{'history':x['history']},'SFT_difficulty_audit',x['trial'],x['session_id']);d=details[(x['task_id'],x['trial'])];good=d['completed'] and d['tool_accurate'] and d['arg_score']==1 and d['ref_score']==1 and d['parse_fail']==0 and d['tool_fail']==0 and d['terminal_semantics_accurate'];assert is_strict_success(m)==good and m.false_finish==d['false_finish'];strict+=int(is_strict_success(m));ff+=int(m.false_finish);per[t['provenance']['family']]+=int(is_strict_success(m))
  eligible=strict<=40
  write(R/'source-difficulty-outcome.json',{'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runs':48,'strict':strict,'false_finish':ff,'per_family_strict_of24':dict(per),'independent_trials_rescored':48,'strict_and_FF_drift':0,'frozen_inputs_and_original_source_unchanged':True,'registered_gate_met':eligible,'next_action':'preregister_matched_curriculum_learning' if eligible else 'close_fixed_difficulty_pilot_no_training','source_model_promoted':False,'candidate_training_started':False,'new_training_steps':0,'evidence_sha256':{str(p):sha(p) for p in (R/'source-new-dev').glob('*') if p.is_file()}})
  state('completed',strict=strict,false_finish=ff,runs=48,gate_met=eligible)
if __name__=='__main__':
 try:main()
 except Exception as e:state('failed',error=str(e));raise
