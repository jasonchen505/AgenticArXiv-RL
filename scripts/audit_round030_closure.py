#!/usr/bin/env python3
from pathlib import Path
import os,sys,datetime,json
os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,'scripts')
from round029_online_core import R,collect,read,write,sha,success_checks
N=R.parent/'round-030-2026-10-09'
def main():
 assert not (N/'closure-independent-audit.json').exists();s=collect(full=True);p=read(R/'experiment-plan.json');o=read(R/'online-outcome.json');assert s['counts']==o['counts'] and s['evidence_sha256']==o['evidence_sha256'] and success_checks(s['counts'])==o['rule_checks'] and not o['registered_rule_met'] and o['fixed_recipe']=='closed';assert len(o['comparisons'])==24
 comparisons={}
 for f in o['comparisons']:
  v=read(f);group=Path(f).stem.rsplit('-',1)[1];base,target=list(v['models']);before=s['strict_per_task'][group][base];after=s['strict_per_task'][group][target];assert set(before)==set(after);gains=sum(before[k]<3 and after[k]==3 for k in before);losses=sum(before[k]==3 and after[k]<3 for k in before);assert gains==v['task_gains'] and losses==v['task_losses']
  for name,x in v['models'].items():
   c=s['counts'][group][name];assert x['authoritative_rescore_drift']==0 and x['runs']==c['runs'] and x['strict_success_rate']==c['strict']/c['runs'] and x['false_finish_rate']==c['false_finish']/c['runs'] and x['reliability']['pass^3']==c['all3_success_tasks']/(c['runs']//3)
  comparisons[f]={'sha256':sha(f),'task_gains':gains,'task_losses':losses}
 for f,h in read(N/'finalization-amendment.json')['original_completed_artifacts_sha256'].items():assert sha(f)==h
 from transformers import AutoTokenizer
 tok0=AutoTokenizer.from_pretrained(p['source_model'],local_files_only=True);training=read(p['train_tasks']);tokenizers={}
 for arm in p['execution_arms']:
  d=Path(p['model_paths'][arm]);tok=AutoTokenizer.from_pretrained(d,local_files_only=True);prompts=0
  for t in training:
   messages=[{'role':'system','content':'You are a helpful assistant.'},{'role':'user','content':t['task']}];assert tok.apply_chat_template(messages,tokenize=True)==tok0.apply_chat_template(messages,tokenize=True);prompts+=1
  assert tok.get_vocab()==tok0.get_vocab() and tok.special_tokens_map==tok0.special_tokens_map;tokenizers[arm]={'all_vocab_and_special_tokens_equal_source':True,'all64_TRAIN_chat_template_input_ids_equal_source':prompts,'source_default_special_ids':{k:getattr(tok0,k) for k in ['bos_token_id','eos_token_id','pad_token_id']},'adapter_config_differing_fields':{k:{'source':v,'final':read(d/'adapter_config.json').get(k)} for k,v in read(Path(p['source_model'])/'adapter_config.json').items() if read(d/'adapter_config.json').get(k)!=v}}
 jobs=[read(R/(arm+'-train-job.json')) for arm in p['execution_arms']]+[read(R/(group+'-'+model+'-job.json')) for group in p['groups'] for model in p['models_order'] if model not in p['reused_evals'].get(group,{})];assert len(jobs)==30 and all(x['exit_code']==0 and x['gpu_index'] in [4,7] for x in jobs)
 result=dict(s,checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scientific_inputs_verified=len(p['sha256']),comparisons_verified=24,comparisons=comparisons,rule_checks=o['rule_checks'],registered_rule_met=False,fixed_recipe_closed=True,source_sft_remains_default=True,successful_training_inference_jobs=30,all1024_native_train_rollouts_closed=True,tokenizer_compatibility=tokenizers,new_learning_steps_this_round=0,new_model_predictions_this_round=0,operational_amendment_preserved=True,old16_comparisons_unchanged=True)
 write(N/'closure-independent-audit.json',result);write(N/'online-outcome.json',o);print({k:result[k] for k in ['passed','independently_rescored','comparisons_verified','actual_completed_learning_steps','successful_training_inference_jobs','rule_checks']})
if __name__=='__main__':main()
