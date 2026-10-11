#!/usr/bin/env python3
"""Matched chosen CE: train-derived program curriculum vs action-matched original replay,795rows199steps."""
from pathlib import Path
import argparse,hashlib,json,math,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'AgenticArxiv'))
from rl import trl_compat
import torch
from datasets import Dataset
from transformers import AutoTokenizer,set_seed
from trl import DPOConfig,DPOTrainer,SFTConfig,SFTTrainer
from trl.trainer.utils import disable_dropout_in_model
from rl.train_grpo import _load_policy
from rl.precision import pin_single_gpu

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['repeat','curriculum'],required=True);p.add_argument('--seed',type=int,choices=[43,44],required=True);p.add_argument('--output',required=True);a=p.parse_args()
    physical_gpu=__import__('os').environ.get('CUDA_VISIBLE_DEVICES');assert physical_gpu in {'4','7'}
    import subprocess
    device=subprocess.check_output(['nvidia-smi',f'--id={physical_gpu}','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
    raw=subprocess.check_output(['nvidia-smi',f'--id={physical_gpu}','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    assert int(device[1].strip())>=23000 and not any(line.split(',')[0].strip()==device[0].strip() for line in raw.splitlines()),'Authorized GPU training capacity changed; no model loaded'
    out=Path(a.output);assert not out.exists();out.mkdir(parents=True)
    source=Path('outputs/repro_20261003/sft_e1/final');data=Path('data/repro_round024')/(('matched_repeat' if a.arm=='repeat' else 'curriculum')+'_combined.jsonl')
    plan=json.loads(Path('artifacts/reproduction/round-024-2026-10-07/experiment-plan.json').read_text());assert sha(data)==plan['sha256'][str(data)];assert sha(source/'adapter_model.safetensors')==plan['sha256'][str(source/'adapter_model.safetensors')]
    manifest=json.loads(Path('artifacts/reproduction/round-024-2026-10-07/matched-data-audit.json').read_text());assert manifest['rows_each']==795
    rows=[json.loads(s) for s in data.read_text().splitlines()]
    tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True);tokenizer.padding_side='right'
    if tokenizer.pad_token is None:tokenizer.pad_token=tokenizer.eos_token
    set_seed(a.seed)
    model=_load_policy(str(source),qlora=True,base_model=None,lora_r=16,lora_alpha=32,lora_dropout=.05)
    disable_dropout_in_model(model)
    common=dict(output_dir=str(out),num_train_epochs=1.,per_device_train_batch_size=1,gradient_accumulation_steps=4,learning_rate=5e-6,max_length=4096,seed=a.seed,data_seed=a.seed,bf16=True,fp16=False,logging_steps=1,logging_first_step=True,save_strategy='no',report_to='none',optim='paged_adamw_8bit',gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},lr_scheduler_type='cosine',warmup_ratio=.03,weight_decay=0.)
    losses=['sft']
    weights=[1.]
    config=DPOConfig(**common,beta=.1,loss_type=losses,loss_weights=weights,precompute_ref_log_probs=False,disable_dropout=True)
    dataset=Dataset.from_list([{k:x[k] for k in ['prompt','chosen','rejected']} for x in rows])
    pin_single_gpu(config);trainer=DPOTrainer(model=model,ref_model=None,args=config,train_dataset=dataset,processing_class=tokenizer)
    state=model.state_dict();pairs=[(k,k.replace('.default.','.ref.')) for k in state if '.default.' in k]
    assert len(pairs)==392 and all(torch.equal(state[k],state[r]) for k,r in pairs)
    ref_before={r:state[r].detach().cpu().clone() for _,r in pairs}
    assert all(not parameter.requires_grad for name,parameter in model.named_parameters() if '.ref.' in name)
    # Exact preprocessed order and index length check in both objectives.
    assert len(trainer.train_dataset)==795
    trainable=[name for name,param in model.named_parameters() if param.requires_grad]
    assert len(trainable)==392 and all('.default.' in name for name in trainable)
    set_seed(a.seed);start=time.monotonic();result=trainer.train();assert trainer.state.global_step==199
    elapsed=time.monotonic()-start
    if ref_before:assert all(torch.equal(model.state_dict()[k].detach().cpu(),v) for k,v in ref_before.items())
    final=out/'final';trainer.save_model(str(final));tokenizer.save_pretrained(final)
    record={'physical_gpu':int(physical_gpu),'objective':'sft','arm':a.arm,'seed':a.seed,'source_model':str(source),'source_adapter_sha256':sha(source/'adapter_model.safetensors'),'preference_data_sha256':sha(data),'actual_steps':trainer.state.global_step,'dataset_rows':len(rows),'batch_size':1,'gradient_accumulation_steps':4,'effective_batch':4,'planned_prompt_exposures':795,'full_dataset_epochs':1,'real_preference_rows':45,'expert_replay_rows':750,'lr':5e-6,'max_length':4096,'max_data_sequence_tokens':manifest['token_lengths']['matched_repeat' if a.arm=='repeat' else 'curriculum']['maximum'],'truncation_count':0,'dropout_disabled':True,'trainable_tensors':len(trainable),'train_loss':result.training_loss,'trainer_elapsed_seconds':elapsed,'torch_peak_gib':torch.cuda.max_memory_allocated()/2**30,'beta':.1,'dpo_loss_type':losses,'dpo_loss_weights':weights,'chosen_only_control_native_ce':True,'reference_equal_sft_and_frozen_tensors':len(ref_before),'sft_completion_only_loss':True,'loss_history':trainer.state.log_history,'data_intervention':'curriculum=398original+352new train-only composition decisions+45unchangederrors;repeat=398original+352original decisions matched by actiontool marginals+45sameerrors. Same795rows199steps, 8new ordered training programs.','caveat':'Native DPOTrainer chosen-token-mean CE only;rejected/reference forwards no learning gradient. Same795rows/199steps,not exact token/FLOP budget. New compositions, argument combinations and longercontexts are the intervention, not pure order-only causality.'}
    assert math.isfinite(record['train_loss'])
    (final/'training_manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k!='loss_history'},indent=2),flush=True)
if __name__=='__main__':main()
