#!/usr/bin/env python3
"""Prepare equal-row, equal-action-marginal curriculum/control data before learning."""
from pathlib import Path
import collections,copy,datetime,hashlib,json
R=Path('artifacts/reproduction/round-024-2026-10-07');D=Path('data/repro_round024')
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();rows=lambda p:[json.loads(x) for x in Path(p).read_text().splitlines()]
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def action(x):
 a=x['chosen'][0]['content'].split('Action: ',1)[1].strip();return 'FINISH' if a=='FINISH' else json.loads(a)['name']
def main():
 assert not (R/'matched-data-plan.json').exists();ind=read(R/'curriculum-independent-audit.json');assert ind['passed']
 orig=Path('data/repro_round014/certified_combined.jsonl');new=D/'curriculum-experts.jsonl';source=rows(orig);curr=rows(new);experts=source[:398];errors=source[398:];assert len(errors)==45 and len(curr)==352
 assert all(x['pair_type']=='expert_replay' and x['chosen']==x['rejected'] for x in experts+curr)
 sources=[orig,new,R/'curriculum-independent-audit.json',Path(__file__),R/'curriculum-preparation-plan.json',R/'baseline-difficulty-plan.json']
 plan={'registered_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'question':'Add train-derived executable long-chain compositions vs repeat original experts matched by added action-type exposure, equalrows/steps. Different argument combinations/longercontexts and taskinstructions remain part of intervention;not pure order-only causalclaim.','rows_each':795,'expert_rows_each':750,'unchanged_error_rows_each':45,'future_steps_each':199,'control':'For each of352curriculum decisions, take the next original expert of the same actiontool(orFINISH), cyclic within frozen original398 order. Original398prefix and same45errortail identical. No sampling, modeloutcomes or bestcases.','source_only_current_diagnostic':'Development source inference mayrun; no outputs used to build/rebalance data. No newlearning until numerical protocol and model/task/data/resource hashes frozen.','seeds_future':[43,44],'learning_permitted':False,'new_training_steps':0,'sha256':{str(p):sha(p) for p in sources}}
 write(R/'matched-data-plan.json',plan)
 pool=collections.defaultdict(list)
 for i,x in enumerate(experts):pool[action(x)].append(i)
 pos=collections.Counter();extra=[];mapping=[]
 for i,x in enumerate(curr):
  name=action(x);j=pool[name][pos[name]%len(pool[name])];pos[name]+=1;extra.append(copy.deepcopy(experts[j]));mapping.append({'added_slot':i,'action_type':name,'original_expert_index':j,'original_semantic_task_id':experts[j]['source_task_id'],'original_trajectory_step':experts[j]['trajectory_step']})
 candidate=experts+curr+errors;control=experts+extra+errors;assert len(candidate)==len(control)==795;assert candidate[:398]==control[:398] and candidate[-45:]==control[-45:];assert collections.Counter(map(action,candidate))==collections.Counter(map(action,control))
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained('outputs/repro_20261003/sft_e1/final',local_files_only=True);lengths={};hashes={}
 for label,data in [('curriculum',candidate),('matched_repeat',control)]:
  ls=[]
  for x in data:
   ids=tok.apply_chat_template(x['prompt']+x['chosen'],tokenize=True);ids=ids['input_ids'] if hasattr(ids,'keys') else ids;ls.append(len(ids))
  assert max(ls)<4096,max(ls);lengths[label]={'maximum':max(ls),'mean':sum(ls)/len(ls),'sequences':len(ls),'truncation':0};p=D/(label+'_combined.jsonl');p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in data));hashes[str(p)]=sha(p)
 # Independent construction from archived mapping checks every row, not just totals.
 saved=rows(D/'matched_repeat_combined.jsonl');assert saved==experts+[experts[m['original_expert_index']] for m in mapping]+errors;assert rows(D/'curriculum_combined.jsonl')==experts+curr+errors
 for p,h in plan['sha256'].items():assert sha(p)==h,p
 write(R/'matched-control-provenance.json',{'mapping':mapping,'exact_same_action_marginals':dict(collections.Counter(map(action,candidate))),'added_action_marginals':dict(pos)})
 write(R/'matched-data-audit.json',{'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'rows_each':795,'steps_each_future':199,'planned_four_arm_steps':796,'planned_row_exposures':3180,'old_expert_prefix_equal_rows':398,'new_expert_rows':352,'unchanged_error_tail_equal_rows':45,'control_all352rows_reconstructed_from_original_experts':True,'exact_action_type_marginals_equal':True,'added_action_marginals':dict(pos),'token_lengths':lengths,'same_row_steps_not_exact_token_compute':True,'sha256':hashes,'source_inputs_unchanged':True,'no_model_outputs_used':True,'new_training_steps':0,'learning_permitted':False})
 print(json.dumps({'rows':795,'future_steps':199,'token_lengths':lengths,'added_marginals':dict(pos)},ensure_ascii=False))
if __name__=='__main__':main()
