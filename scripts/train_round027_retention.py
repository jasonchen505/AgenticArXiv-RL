#!/usr/bin/env python3
from pathlib import Path
import argparse,hashlib,json,math,os,subprocess,sys,time
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1]/'AgenticArxiv')]
from rl import trl_compat
import torch
from datasets import Dataset
from transformers import AutoTokenizer,set_seed
from trl import DPOConfig
from trl.trainer.utils import disable_dropout_in_model
from rl.train_grpo import _load_policy
from rl.precision import pin_single_gpu
from round027_retention_core import RetentionTrainer,encode_anchor
R=Path('artifacts/reproduction/round-027-2026-10-08');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def admission():
 gpu=os.environ.get('CUDA_VISIBLE_DEVICES');assert gpu in {'4','7'}
 uuid,free=[x.strip() for x in subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')]
 processes=subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 assert int(free)>=23000 and not any(x.split(',')[0].strip()==uuid for x in processes.splitlines())
 return int(gpu)
def construct(arm,seed,out):
 source=Path('outputs/repro_20261003/sft_e1/final');data=Path('data/repro_round024')/(('matched_repeat' if arm=='repeat' else 'curriculum')+'_combined.jsonl');rows=[json.loads(x) for x in data.read_text().splitlines()]
 tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True);tokenizer.padding_side='right'
 if tokenizer.pad_token is None:tokenizer.pad_token=tokenizer.eos_token
 set_seed(seed);model=_load_policy(str(source),qlora=True,base_model=None,lora_r=16,lora_alpha=32,lora_dropout=.05);disable_dropout_in_model(model)
 config=DPOConfig(output_dir=str(out),num_train_epochs=1.,per_device_train_batch_size=1,gradient_accumulation_steps=4,learning_rate=5e-6,max_length=4096,seed=seed,data_seed=seed,bf16=True,fp16=False,logging_steps=1,logging_first_step=True,save_strategy='no',report_to='none',optim='paged_adamw_8bit',gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},lr_scheduler_type='cosine',warmup_ratio=.03,weight_decay=0.,beta=.1,loss_type=['sft'],loss_weights=[1.],precompute_ref_log_probs=False,disable_dropout=True)
 pin_single_gpu(config)
 anchors=[encode_anchor(tokenizer,json.loads(x)) for x in Path('data/repro_round026/source-train-anchors.jsonl').read_text().splitlines()];assert len(anchors)==443
 dataset=Dataset.from_list([{k:x[k] for k in ['prompt','chosen','rejected']} for x in rows])
 trainer=RetentionTrainer(model=model,ref_model=None,args=config,train_dataset=dataset,processing_class=tokenizer,anchors=anchors,anchor_seed=seed,retention_weight=.1)
 assert len(trainer.train_dataset)==795
 return trainer,tokenizer,model,data
def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['repeat','curriculum'],required=True);p.add_argument('--seed',type=int,choices=[43,44],required=True);p.add_argument('--output',required=True);args=p.parse_args();gpu=admission();out=Path(args.output);assert not out.exists()
 plan=json.loads((R/'experiment-plan.json').read_text());assert plan['learning_permitted'] and plan['coefficient']==.1
 for path,h in plan['sha256'].items():assert sha(path)==h,path
 out.mkdir(parents=True);trainer,tok,model,data=construct(args.arm,args.seed,out)
 state=model.state_dict();pairs=[(k,k.replace('.default.','.ref.')) for k in state if '.default.' in k];assert len(pairs)==392 and all(torch.equal(state[k],state[r]) for k,r in pairs)
 ref_before={r:state[r].detach().cpu().clone() for _,r in pairs};policy_before={k:state[k].detach().cpu().clone() for k,_ in pairs}
 assert all(not p.requires_grad for n,p in model.named_parameters() if '.ref.' in n)
 set_seed(args.seed);start=time.monotonic();result=trainer.train();assert trainer.state.global_step==199 and trainer.anchor_calls==795
 assert all(torch.equal(model.state_dict()[k].detach().cpu(),v) for k,v in ref_before.items());changed=sum(not torch.equal(model.state_dict()[k].detach().cpu(),v) for k,v in policy_before.items());assert changed==392
 assert [x['anchor_index'] for x in trainer.anchor_logs]==plan['anchor_orders'][str(args.seed)]
 assert all(x['accumulation_group_size']==4 for x in trainer.anchor_logs[:-3]) and all(x['accumulation_group_size']==3 for x in trainer.anchor_logs[-3:])
 final=out/'final';trainer.save_model(str(final));tok.save_pretrained(final)
 record={'physical_gpu':gpu,'arm':args.arm,'seed':args.seed,'dataset_rows':795,'actual_steps':199,'source_model':plan['source_model'],'data_sha256':sha(data),'source_adapter_sha256':plan['sha256']['outputs/repro_20261003/sft_e1/final/adapter_model.safetensors'],'objective':'native chosen CE +0.1 mean-completion full-vocabulary KL(sourceSFT||policy) on originalTRAIN anchors','loss_type':['sft'],'retention_coefficient':.1,'anchor_forward_backward_calls':795,'anchor_unique_rows':443,'anchor_exposures_order':trainer.anchor_order,'anchor_completion_tokens':sum(x['completion_tokens'] for x in trainer.anchor_logs),'reference_equal_source_tensors':392,'changed_adapter_tensors':changed,'anchor_logs':trainer.anchor_logs,'loss_history':trainer.state.log_history,'train_loss':result.training_loss,'trainer_elapsed_seconds':time.monotonic()-start,'torch_peak_gib':torch.cuda.max_memory_allocated()/2**30,'native_ce_recipe_unchanged':True,'separate_backwards_same_sum_gradient_before_clip':True,'additional_anchor_compute_not_budget_matched_to_historical_CE':True}
 assert math.isfinite(record['train_loss']);(final/'training_manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k not in ['loss_history','anchor_logs','anchor_exposures_order']},indent=2),flush=True)
if __name__=='__main__':main()
