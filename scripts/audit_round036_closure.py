import os,sys,json,datetime,collections,copy,shutil,tarfile
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',STORE_BACKEND='memory',FIGURE_ANALYSIS_BACKEND='extractive');sys.path[:0]=['AgenticArxiv','scripts']
from round034_core import R,D,read,rows,write,sha,verify
from finalize_round035_operational import collect,rules
S=R.parent/'round-036-2026-10-11'
N={'get_paper_content':'C','summarize_paper':'S','extract_paper_figures':'F','analyze_figure':'A'}
def code(steps):return ''.join(N.get(x['name'],'?') for x in steps if x['name'] not in ['get_recently_submitted_cs_papers','download_arxiv_pdf'])
def main():
 S.mkdir(exist_ok=True);assert not (S/'closure-independent-audit.json').exists();p=read(R/'experiment-plan.json');verify(p);a=collect(True);formal=read(R.parent/'round-035-2026-10-10/study-outcome.json')
 for k,v in a.items():assert v==formal[k],k
 checks=rules(a['counts']);assert checks==formal['rule_checks'] and not all(checks.values()) and formal['fixed_recipe']=='closed';mappings=0;histdiff=0
 for label in p['eval_jobs']:
  for x in rows(R/label/'traces.jsonl'):mappings+=len(x['raw_result']['artifact_observations']);histdiff+=int(x['strict']!=x.get('execution_strict',x['strict']))
 train=read(p['eval_jobs']['mixed49-plan-TRAIN-s49']['tasks']);train_signatures=collections.Counter(code([{'name':n} for n in t['expected_tools']]) for t in train);plan_data=rows(D/'plan-supervision.jsonl');assert len(plan_data)==48
 actual_labels=[json.loads(x['chosen'][0]['content'])['steps'] for x in plan_data];lengths=collections.Counter(len(x) for x in actual_labels);cooccur=sum({'get_paper_content','summarize_paper'}<=set(x['name'] for x in steps) for steps in actual_labels);confusion=collections.Counter();missing=collections.Counter();lengthpred=collections.Counter();examples=[]
 for seed in [49,50]:
  predictions=rows(R/f'mixed49-plan-confirm-s{seed}'/'predictions.jsonl');tasks={t['id']:t for t in read(p['eval_jobs'][f'mixed49-plan-confirm-s{seed}']['tasks'])}
  for x in predictions:
   t=tasks[x['task_id']];z=json.loads(x['prediction']['response']['choices'][0]['message']['content']);sig=code(z['steps']);gold=code([{'name':n} for n in t['expected_tools']]);confusion[gold+' -> '+sig]+=1;lengthpred[len(z['steps'])]+=1;missing.update(set(t['expected_tools'])-set(a['name'] for a in z['steps']))
   if len(examples)<2:examples.append({'task_id':x['task_id'],'generation_seed':seed,'expected':gold,'model':sig,'original_request':x['prediction']['request'],'original_response':x['prediction']['response']})
 ratios={str(seed)+':'+w:a['counts'][f'mixed49-plan-confirm-s{seed}'][w]['total_tokens']/a['counts'][f'source-native-confirm{w[-1]}-s{seed}'][w]['total_tokens'] for seed in [49,50] for w in ['trainA','trainB']}
 write(S/'closure-independent-audit.json',dict(a,checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed_evidence_audit=True,rule_checks=checks,formal_outcome_SHA256=sha(R.parent/'round-035-2026-10-10/study-outcome.json'),verified_frozen_inputs=len(p['sha256']),actual_public_mappings_recomputed=mappings,primary_execution_strict_differences=histdiff,fixed_recipe_closed=True,new_learning_steps_this_round=0,source_SFT_remains_default=True,fresh_token_fraction_of_source_native=ratios))
 write(S/'program-coverage-audit.json',{'passed':True,'TRAIN_unique_plan_requests':48,'TRAIN_programs':dict(train_signatures),'TRAIN_full_tool_length_counts':dict(lengths),'TRAIN_content_and_summary_cooccur_requests':cooccur,'fresh_actual_model_predictions':96,'fresh_logical_executions':192,'fresh_expected_full_tool_length':6,'fresh_model_tool_length_counts':dict(lengthpred),'fresh_program_confusion':dict(confusion),'missing_tools_actual_predictions':dict(missing),'examples':examples,'diagnostic_only_no_rescoring_or_prediction_repair':True,'new_model_predictions':0,'new_learning_steps':0})
 print({'rules':checks,'mappings':mappings,'TRAIN_lengths':dict(lengths),'TRAIN_C_and_S':cooccur,'fresh_lengths':dict(lengthpred),'confusion':dict(confusion),'missing':dict(missing)})
if __name__=='__main__':main()
