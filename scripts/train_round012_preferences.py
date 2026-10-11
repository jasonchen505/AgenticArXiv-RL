#!/usr/bin/env python3
"""Chosen-likelihood anchored DPO; frozen Round010 data and settings."""
from pathlib import Path
import argparse,hashlib,json,math,os,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'AgenticArxiv'))
from rl import trl_compat
import torch
from datasets import Dataset
from transformers import AutoTokenizer,set_seed
from trl import DPOConfig,DPOTrainer
from trl.trainer.utils import disable_dropout_in_model
from rl.train_grpo import _load_policy
from rl.precision import pin_single_gpu

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--objective',choices=['dpo_sft'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='4','Training must be launched on physical GPU4 only'
    out=Path(a.output);assert not out.exists();out.mkdir(parents=True)
    source=Path('outputs/repro_20261003/sft_e1/final');data=Path('data/repro_round010/preferences.jsonl')
    manifest=json.loads(Path('artifacts/reproduction/round-010-2026-10-04/preference-data-manifest.json').read_text());assert sha(data)==manifest['sha256'][str(data)]
    rows=[json.loads(s) for s in data.read_text().splitlines()]
    tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True);tokenizer.padding_side='right'
    if tokenizer.pad_token is None:tokenizer.pad_token=tokenizer.eos_token
    set_seed(a.seed)
    model=_load_policy(str(source),qlora=True,base_model=None,lora_r=16,lora_alpha=32,lora_dropout=.05)
    disable_dropout_in_model(model)
    common=dict(output_dir=str(out),max_steps=51,per_device_train_batch_size=1,gradient_accumulation_steps=4,learning_rate=5e-6,max_length=4096,seed=a.seed,data_seed=a.seed,bf16=True,fp16=False,logging_steps=1,logging_first_step=True,save_strategy='no',report_to='none',optim='paged_adamw_8bit',gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},lr_scheduler_type='cosine',warmup_ratio=.03,weight_decay=0.)
    if a.objective=='dpo_sft':
        config=DPOConfig(**common,beta=.1,loss_type=['sigmoid','sft'],loss_weights=[1.0,1.0],precompute_ref_log_probs=False,disable_dropout=True)
        dataset=Dataset.from_list([{k:x[k] for k in ['prompt','chosen','rejected']} for x in rows])
        pin_single_gpu(config);trainer=DPOTrainer(model=model,ref_model=None,args=config,train_dataset=dataset,processing_class=tokenizer)
        state=model.state_dict();pairs=[(k,k.replace('.default.','.ref.')) for k in state if '.default.' in k]
        assert len(pairs)==392 and all(torch.equal(state[k],state[r]) for k,r in pairs)
        ref_before={r:state[r].detach().cpu().clone() for _,r in pairs}
        assert all(not parameter.requires_grad for name,parameter in model.named_parameters() if '.ref.' in name)
    # Exact preprocessed order and index length check in both objectives.
    assert len(trainer.train_dataset)==448
    trainable=[name for name,param in model.named_parameters() if param.requires_grad]
    assert len(trainable)==392 and all('.default.' in name for name in trainable)
    set_seed(a.seed);start=time.monotonic();result=trainer.train();assert trainer.state.global_step==51
    elapsed=time.monotonic()-start
    if ref_before:assert all(torch.equal(model.state_dict()[k].detach().cpu(),v) for k,v in ref_before.items())
    final=out/'final';trainer.save_model(str(final));tokenizer.save_pretrained(final)
    record={'objective':a.objective,'seed':a.seed,'source_model':str(source),'source_adapter_sha256':sha(source/'adapter_model.safetensors'),'preference_data_sha256':sha(data),'actual_steps':trainer.state.global_step,'dataset_rows':len(rows),'batch_size':1,'gradient_accumulation_steps':4,'effective_batch':4,'planned_prompt_exposures':204,'lr':5e-6,'max_length':4096,'max_data_sequence_tokens':manifest['max_sequence_tokens'],'truncation_count':0,'dropout_disabled':True,'trainable_tensors':len(trainable),'train_loss':result.training_loss,'trainer_elapsed_seconds':elapsed,'torch_peak_gib':torch.cuda.max_memory_allocated()/2**30,'beta':.1 if a.objective=='dpo_sft' else None,'dpo_loss_type':['sigmoid','sft'] if a.objective=='dpo_sft' else None,'dpo_loss_weights':[1.0,1.0],'chosen_sft_loss':'Token-mean completion-only cross entropy, native TRL sft loss branch','reference_equal_sft_and_frozen_tensors':len(ref_before),'chosen_sft_completion_only_loss':True,'loss_history':trainer.state.log_history,'caveat':'Same steps/effective prompt batch/source/chosen data and sequence bound, not equal FLOPs or supervised tokens. Synthetic train-only negatives.'}
    assert math.isfinite(record['train_loss'])
    (final/'training_manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k!='loss_history'},indent=2),flush=True)
if __name__=='__main__':main()
