#!/usr/bin/env python3
"""Independently rescore/reconstruct train-only curriculum and measure actual coverage."""
from pathlib import Path
import collections,datetime,hashlib,json,os,sys,tempfile
R=Path('artifacts/reproduction/round-024-2026-10-07');D=Path('data/repro_round024');OLD=R.parent/'round-022-2026-10-07'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();rows=lambda p:[json.loads(x) for x in Path(p).read_text().splitlines()]
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 assert not (R/'curriculum-independent-audit.json').exists();plan=read(R/'curriculum-preparation-plan.json');validation=read(R/'curriculum-gold-validation.json');admission=read(R/'curriculum-resource-admission.json')
 for p,h in dict(plan['sha256'],**plan['asset_sha256'],**admission['sha256']).items():assert sha(p)==h,p
 for perm in admission['permutations']:assert sorted(perm['before'])==sorted(perm['after']) and perm['membership_preserved']
 os.environ.update(CUDA_VISIBLE_DEVICES='',STORE_BACKEND='memory');sys.path.insert(0,'AgenticArxiv')
 from benchmark.metrics import extract_metrics,is_strict_success
 from agents.prompt_templates import get_react_prompt,build_visible_setup_context
 from transformers import AutoTokenizer
 source=rows('data/repro_round014/certified_combined.jsonl');desc=source[0]['prompt'][0]['content'].split('你有以下工具可以使用：\n\n',1)[1].split('\n\n当前任务：',1)[0]
 original=read(R.parent/'round-013-2026-10-05/collection-oracles.json');known=read(R.parent/'round-002-2026-10-02/task-oracles.json');known_programs={tuple(t['expected_tools']) for t in original+known};train_programs={tuple(t['expected_tools']) for t in original};domains=collections.defaultdict(set)
 for t in original:
  for name,a in zip(t['expected_tools'],t['expected_tool_args']):
   for k,v in a.items():domains[(name,k)].add(json.dumps(v,sort_keys=True))
 programs={};traces={};allgold=0;resultsha={};alltasks=[]
 for split,v in plan['splits'].items():
  ts=read(R/(split+'-oracles.json'));gold=rows(R/(split+'-gold-traces.jsonl'));assert len(ts)==len(gold)==v['tasks'];td={t['id']:t for t in ts};assert set(td)=={x['task_id'] for x in gold}
  programs[split]={tuple(t['expected_tools']) for t in ts};assert not programs[split]&known_programs;alltasks+=ts
  assert len(programs[split])==v['families']
  for t in ts:
   for name,a in zip(t['expected_tools'],t['expected_tool_args']):
    for k,val in a.items():assert json.dumps(val,sort_keys=True) in domains[(name,k)],(name,k,val)
  for x in gold:
   t=td[x['task_id']];m=extract_metrics(t,{'history':x['history']},'independent_CPU_curriculum',0,'');assert is_strict_success(m) and not m.false_finish;assert x['metrics']['false_finish']==m.false_finish
   assert len(x['history'])==len(t['expected_tools'])+1 and x['history'][-1]['action']=='FINISH'
   for h,name,args in zip(x['history'][:-1],t['expected_tools'],t['expected_tool_args']):assert json.loads(h['action'])=={'name':name,'args':args}
   traces[x['task_id']]=(t,x['history']);allgold+=1
  for kind in ['tasks.json','oracles.json','gold-traces.jsonl']:p=R/(split+'-'+kind);resultsha[str(p)]=sha(p)
 assert allgold==112;assert not programs['train']&programs['dev'] and not programs['train']&programs['confirmA'] and not programs['dev']&programs['confirmA'];assert programs['confirmA']==programs['confirmB']
 data=rows(D/'curriculum-experts.jsonl');assert len(data)==352 and len({x['source_task_id'] for x in data})==64;assert sha(D/'curriculum-experts.jsonl')==validation['curriculum_data_sha256'];seen=set()
 for x in data:
  t,hist=traces[x['source_task_id']];i=x['trajectory_step'];assert t['provenance']['family'] in plan['families']['train'] and x['parent_task_id']==t['parent_task_id']
  assert (x['source_task_id'],i) not in seen;seen.add((x['source_task_id'],i));h=hist[i]
  assert x['chosen']==x['rejected']==[{'role':'assistant','content':'Thought: '+h['thought']+'\nAction: '+h['action']}]
  prefix='\n\n'.join('Thought: '+h['thought']+'\nAction: '+h['action']+'\nObservation: '+h['observation'] for h in hist[:i]);initial=build_visible_setup_context(t).strip();context='\n\n'.join(p for p in [initial,prefix] if p)
  assert x['prompt']==[{'role':'user','content':get_react_prompt(t['task'],desc,context)}]
 assert seen=={(t['id'],i) for t in alltasks if t['provenance']['family'] in plan['families']['train'] for i in range(len(t['expected_tools'])+1)}
 tok=AutoTokenizer.from_pretrained('outputs/repro_20261003/sft_e1/final',local_files_only=True);ls=[]
 for x in data:
  ids=tok.apply_chat_template(x['prompt']+x['chosen'],tokenize=True);ids=ids['input_ids'] if hasattr(ids,'keys') else ids;ls.append(len(ids))
 assert max(ls)<4096 and validation['token_lengths']=={'maximum':max(ls),'mean':sum(ls)/len(ls),'sequences':len(ls),'truncation':0}
 assert len(train_programs)==19 and len(train_programs|programs['train'])==27
 coverage={}
 for label in ['repeat','world']:
  experts=[x for x in rows('data/repro_round022_v2/'+label+'_combined.jsonl') if x['pair_type']=='expert_replay'];raw=set();normalized=set();hist=collections.Counter()
  for x in experts:
   a=x['chosen'][0]['content'].split('Action: ',1)[1].strip()
   if a=='FINISH':raw.add(a);normalized.add(a);hist[a]+=1;continue
   obj=json.loads(a);raw.add(json.dumps(obj,sort_keys=True,ensure_ascii=False));hist[obj['name']]+=1
   if isinstance(obj['args'].get('ref'),str):obj['args']['ref']='<literal-paper-reference>'
   normalized.add(json.dumps(obj,sort_keys=True,ensure_ascii=False))
  coverage[label]={'expert_rows':len(experts),'train_parents':len({x['parent_task_id'] for x in experts}),'semantic_tasks':len({x['source_task_id'] for x in experts}),'ordered_programs':19,'raw_unique_actions_including_FINISH':len(raw),'literal_normalized_unique_actions_including_FINISH':len(normalized),'action_row_counts':dict(hist)}
 assert coverage['repeat']['action_row_counts']==coverage['world']['action_row_counts'];assert coverage['repeat']['literal_normalized_unique_actions_including_FINISH']==coverage['world']['literal_normalized_unique_actions_including_FINISH']
 write(R/'decision-coverage-audit.json',{'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'datasets':coverage,'intervention':{'old_train_ordered_programs':19,'added_train_ordered_programs':8,'after_adding_candidate':27,'dev_ordered_programs':2,'confirmation_ordered_programs':2,'full_sequence_overlap_with_old_train_and_benchmark':0,'full_sequence_overlap_train_dev_confirm':0},'interpretation':'More paper observations did not add numeric/null actions or ordered programs. This motivates, but does not establish, a program-coverage bottleneck. Raw extra10actions are entity-literal retargets. New families share tools/parameters/subsequences; only full ordered signatures disjoint. 16tasks mean2families*8parameters, not16 independent programs.','GPU_jobs':0,'new_training_steps':0})
 resultsha[str(D/'curriculum-experts.jsonl')]=sha(D/'curriculum-experts.jsonl')
 write(R/'curriculum-independent-audit.json',{'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'raw_gold_independently_rescored':112,'strict_and_FF_drift':0,'train_rows_reconstructed':352,'training_task_ids':64,'all_generation_parameter_values_present_in_train':True,'full_program_overlap_original_and_prior_benchmark':0,'full_program_overlap_train_dev_confirm':0,'confirm_worlds_share_same2programs':True,'old_ordered_programs':19,'after_adding_train_curriculum_programs':27,'source_and_asset_hashes_checked':len(set(plan['sha256'])|set(plan['asset_sha256'])|set(admission['sha256'])),'sources_and_real_assets_unchanged':True,'token_lengths':validation['token_lengths'],'data_sha256':resultsha,'GPU_jobs':0,'new_training_steps':0,'learning_permitted':False})
 print(json.dumps({'gold':allgold,'train_rows':len(data),'programs':27,'maximum_tokens':max(ls),'coverage':coverage},ensure_ascii=False))
if __name__=='__main__':main()
