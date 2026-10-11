#!/usr/bin/env python3
"""CPU-only independent closure; don't rewrite Round028 report snapshots or finalizer outputs."""
from pathlib import Path
import os,sys,json,collections,statistics,hashlib,datetime
os.environ['CUDA_VISIBLE_DEVICES']='';sys.path[:0]=['AgenticArxiv','scripts']
R=Path('artifacts/reproduction/round-029-2026-10-09');OLD=R.parent/'round-028-2026-10-09';OUT=Path('outputs/repro_round028/program_probe')
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def main():
 R.mkdir(exist_ok=True);assert not (R/'probe-independent-audit.json').exists();plan=read(OLD/'program-probe-plan.json');out=read(OLD/'program-probe-outcome.json');ann=read(OLD/'probe-before-model-annex.json');job=read(OLD/'program-probe-job.json')
 for p,h in plan['sha256'].items():assert sha(p)==h,p
 assert sha(ann['execution_source'])==ann['execution_source_sha256'] and sha(OLD/'program-probe-plan.json')==ann['original_plan_sha256'] and job['exit_code']==0 and job['gpu_index'] in [4,7]
 from safetensors.torch import load_file
 import torch
 source=load_file(plan['source_model']+'/adapter_model.safetensors',device='cpu');policy=load_file(str(OUT/'final/adapter_model.safetensors'),device='cpu');ref=load_file(str(OUT/'final/ref/adapter_model.safetensors'),device='cpu');assert set(source)==set(policy)==set(ref) and len(source)==392 and all(torch.equal(source[k],policy[k]) and torch.equal(source[k],ref[k]) for k in source)
 manifest=read(OUT/'final/training_manifest.json');assert manifest['actual_steps']==64 and manifest['learning_rate']==0 and manifest['frozen_policy_probe']
 from rl.reward import RewardCalculator
 from benchmark.metrics import is_strict_success
 calc=RewardCalculator(curriculum_steps=0);tasks={x['id']:x for x in read(plan['oracles'])};raw=[json.loads(l) for l in (OUT/'rollout_traces.jsonl').read_text().splitlines()];assert len(raw)==256 and collections.Counter(x['task_id'] for x in raw)==dict.fromkeys(tasks,4)
 groups=collections.defaultdict(list);family=collections.defaultdict(collections.Counter);counts=collections.Counter();scored=[]
 for x in raw:
  t=tasks[x['task_id']];b,m=calc.compute_reward_breakdown(t,x['trajectory']);assert abs(b.total-x['reward'])<1e-5;ok=is_strict_success(m);ff=bool(m.false_finish);clip=bool(x['trajectory'].get('clipped',False));counts.update(strict=int(ok),false_finish=int(ff),clipped=int(clip),full_reward_non_strict=int(not ok and x['reward']>=1-1e-6));family[t['provenance']['family']].update(samples=1,strict=int(ok),false_finish=int(ff))
  z={'task_id':x['task_id'],'training_step':x['training_step'],'batch_index':x['batch_index'],'group_index':x['group_index'],'generation_index':x['generation_index'],'strict':ok,'false_finish':ff,'reward':x['reward']};scored.append(z);groups[x['task_id'],x['training_step'],x['batch_index'],x['group_index']].append(z)
 assert len(groups)==64 and all(len(v)==4 and {x['generation_index'] for x in v}=={0,1,2,3} for v in groups.values())
 details=[];mixedcontracts=set();pairs=inversions=ties=0
 for k,v in groups.items():
  success=sum(x['strict'] for x in v);variable=statistics.pstdev(x['reward'] for x in v)>1e-8;is_mixed=0<success<4;t=tasks[k[0]]
  if is_mixed:mixedcontracts.add(json.dumps([t['task'],t['expected_tools'],t['expected_tool_args']],sort_keys=True,ensure_ascii=False))
  for yes in [x for x in v if x['strict']]:
   for no in [x for x in v if not x['strict']]:pairs+=1;inversions+=int(no['reward']>yes['reward']+1e-8);ties+=int(abs(no['reward']-yes['reward'])<=1e-8)
  details.append({'task_id':k[0],'family':t['provenance']['family'],'strict':success,'reward_variance_nonzero':variable,'reward_values':[x['reward'] for x in v]})
 mixed=sum(0<x['strict']<4 for x in details);var=sum(x['reward_variance_nonzero'] for x in details);checks={'complete256_and64_groups':True,'mixed_groups_at_least16':mixed>=16,'reward_variable_groups_at_least16':var>=16,'clipped_samples_at_most4':counts['clipped']<=4,'mixed_unique_contracts_at_least12':len(mixedcontracts)>=12,'all392_policy_and_reference_equal_source':True}
 assert out['actual_rollouts']==256 and out['strict_successes']==counts['strict'] and out['mixed_strict_groups']==mixed and out['mixed_unique_visible_contracts']==len(mixedcontracts) and out['nonzero_legacy_variance_groups']==var and out['token_clipped_samples']==counts['clipped'] and out['eligibility_gate_met']==all(checks.values()) and sha(OUT/'rollout_traces.jsonl')==out['raw_traces_sha256']
 contracts=collections.Counter(json.dumps([t['task'],t['expected_tools'],t['expected_tool_args']],sort_keys=True,ensure_ascii=False) for t in tasks.values());assert len(contracts)==48 and collections.Counter(contracts.values())=={1:32,2:16}
 result={'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_only':True,'passed':True,'all_samples_independently_rescored':256,'strict_and_reward_drift':0,'counts':dict(counts),'groups':64,'mixed_strict_groups':mixed,'mixed_unique_visible_contracts':len(mixedcontracts),'nonzero_reward_variance_groups':var,'all4_strict_groups':sum(x['strict']==4 for x in details),'all0_strict_groups':sum(x['strict']==0 for x in details),'visible_contracts':48,'nominal_ids':64,'family_counts':dict(family),'within_group_strict_success_failure_reward_pairs':pairs,'reward_inversion_pairs':inversions,'reward_tie_pairs':ties,'eligibility_checks':checks,'eligibility_gate_met':all(checks.values()),'policy_and_reference_source_equal_tensors':392,'native_lr0_optimizer_loop_steps':64,'learning_updates':0,'scientific_inputs_verified':len(plan['sha256']),'no_effectiveness_claim':True,'source_sft_remains_default':True,'details':details,'sha256':{str(p):sha(p) for p in [OLD/'program-probe-plan.json',OLD/'probe-before-model-annex.json',OLD/'program-probe-outcome.json',OLD/'program-probe-job.json',OUT/'rollout_traces.jsonl',OUT/'final/training_manifest.json',Path(__file__)]}}
 write(R/'probe-independent-audit.json',result);write(R/'program-probe-outcome.json',out);(R/'probe-rescored.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in scored));print(json.dumps({k:result[k] for k in ['passed','counts','mixed_strict_groups','mixed_unique_visible_contracts','nonzero_reward_variance_groups','within_group_strict_success_failure_reward_pairs','reward_inversion_pairs','reward_tie_pairs','eligibility_gate_met']}))
if __name__=='__main__':main()
