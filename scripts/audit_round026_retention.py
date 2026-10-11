#!/usr/bin/env python3
"""Independently reconstruct source-only anchors and rescore prospective CPU gold."""
from pathlib import Path
import collections,datetime,hashlib,itertools,json,os,sys
R=Path('artifacts/reproduction/round-026-2026-10-08');OLD=R.parent/'round-024-2026-10-07';D=Path('data/repro_round026')
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def main():
 assert not (R/'retention-independent-audit.json').exists();os.environ.update(CUDA_VISIBLE_DEVICES='',STORE_BACKEND='memory');sys.path.insert(0,'AgenticArxiv')
 from benchmark.metrics import extract_metrics,is_strict_success
 p=read(R/'retention-preparation-plan.json');result=read(R/'retention-preparation-result.json');original=rows('data/repro_round014/certified_combined.jsonl');anchors=rows(D/'source-train-anchors.jsonl');train={t['id']:t for t in read(R.parent/'round-013-2026-10-05/collection-oracles.json')}
 for path,h in dict(p['scientific_inputs_sha256'],**p['source_sha256'],**result['files_sha256']).items():assert sha(path)==h,path
 assert len(anchors)==443
 for i,(a,x) in enumerate(zip(anchors,original)):
  assert a=={'anchor_index':i,'source_task_id':x['source_task_id'],'pair_type':x['pair_type'],'terminal_mode':train[x['source_task_id']].get('expected_terminal_mode','completed'),'prompt':x['prompt'],'chosen':x['chosen']}
 excluded={tuple(t['expected_tools']) for t in list(train.values())+read(R.parent/'round-002-2026-10-02/task-oracles.json')+read(OLD/'all-tasks.json')};names={'C':'get_paper_content','S':'summarize_paper','F':'extract_paper_figures','A':'analyze_figure'};sig=lambda f:tuple(['get_recently_submitted_cs_papers','download_arxiv_pdf']+[names[c] for c in f])
 candidates=sorted(''.join(x) for x in itertools.permutations('CSFA') if x.index('F')<x.index('A'));fs=[f for f in candidates if sig(f) not in excluded][:2];assert fs==p['families']==['CFAS','CFSA']
 domains=collections.defaultdict(set)
 for t in train.values():
  for name,args in zip(t['expected_tools'],t['expected_tool_args']):
   for k,v in args.items():domains[name,k].add(json.dumps(v,sort_keys=True))
 tasks={};taskhash={}
 for split in ['confirmA','confirmB']:
  path=R/(split+'-oracles.json');ts=read(path);assert len(ts)==16 and collections.Counter(t['provenance']['family'] for t in ts)=={f:8 for f in fs}
  assert {tuple(t['expected_tools']) for t in ts}=={sig(f) for f in fs}
  for t in ts:
   assert t['id'] not in train and tuple(t['expected_tools']) not in excluded
   for n,args in zip(t['expected_tools'],t['expected_tool_args']):
    for k,v in args.items():assert json.dumps(v,sort_keys=True) in domains[n,k]
  tasks.update({t['id']:t for t in ts});taskhash[str(path)]=sha(path)
 gold=rows(R/'confirmation-gold-traces.jsonl');assert len(gold)==32 and collections.Counter(x['task_id'] for x in gold)==dict.fromkeys(tasks,1)
 for x in gold:
  t=tasks[x['task_id']];m=extract_metrics(t,{'history':x['history']},'CPU_R026_second_scoring',0,'independent_'+t['id']);assert is_strict_success(m) and not m.false_finish
  stored=x['metrics'];now=m.to_dict()
  for k in ['completed','tool_accurate','arg_score','ref_score','parse_fail','tool_fail','terminal_semantics_accurate','false_finish']:assert stored[k]==now[k]
 assert p['learning_permitted']==result['learning_permitted']==False and not list(R.glob('*-job.json')) and not Path('outputs/repro_round026').exists()
 out={'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'CPU_only':True,'all443anchors_exact_original_train_reconstructed':True,'anchor_rows':443,'gold_independently_rescored':32,'strict_and_FF_drift':0,'families':fs,'new_confirmation_programs_not_in_all_prior_splits':True,'all_argument_domains_original_train':True,'all1215_scientific_inputs_unchanged':True,'no_heldout_trace_in_anchors':True,'no_model_prediction_or_learning':True,'new_training_steps':0,'learning_permitted':False,'future_trainer_and_full_numeric_learning_protocol_pending':True,'files_sha256':result['files_sha256'],'preparation_plan_sha256':sha(R/'retention-preparation-plan.json'),'anchor_terminal_modes':dict(collections.Counter(a['terminal_mode'] for a in anchors))}
 (R/'retention-independent-audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['files_sha256']},ensure_ascii=False))
if __name__=='__main__':main()
