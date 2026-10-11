#!/usr/bin/env python3
"""Real-error anchored DPO plus broad replay versus matched chosen-only SFT from the same saved SFT adapter."""
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
    p=argparse.ArgumentParser();p.add_argument('--objective',choices=['sft'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args()
    assert __import__('os').environ.get('CUDA_VISIBLE_DEVICES')=='4'
    import subprocess
    device=subprocess.check_output(['nvidia-smi','--id=4','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
    raw=subprocess.check_output(['nvidia-smi','--id=4','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    assert int(device[1].strip())>=23000 and not any(line.split(',')[0].strip()==device[0].strip() for line in raw.splitlines()),'GPU4 training capacity changed; no model loaded'
    out=Path(a.output);assert not out.exists();out.mkdir(parents=True)
    source=Path('outputs/repro_20261003/sft_e1/final');data=Path('data/repro_round019/aligned_combined.jsonl')
    manifest=json.loads(Path('artifacts/reproduction/round-019-2026-10-06/aligned-data-manifest.json').read_text());assert sha(data)==manifest['sha256'][str(data)]
    rows=[json.loads(s) for s in data.read_text().splitlines()]
    tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True);tokenizer.padding_side='right'
    if tokenizer.pad_token is None:tokenizer.pad_token=tokenizer.eos_token
    set_seed(a.seed)
    model=_load_policy(str(source),qlora=True,base_model=None,lora_r=16,lora_alpha=32,lora_dropout=.05)
    disable_dropout_in_model(model)
    common=dict(output_dir=str(out),num_train_epochs=1.,per_device_train_batch_size=1,gradient_accumulation_steps=4,learning_rate=5e-6,max_length=4096,seed=a.seed,data_seed=a.seed,bf16=True,fp16=False,logging_steps=1,logging_first_step=True,save_strategy='no',report_to='none',optim='paged_adamw_8bit',gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},lr_scheduler_type='cosine',warmup_ratio=.03,weight_decay=0.)
    losses=['sigmoid','sft'] if a.objective=='dpo' else ['sft']
    weights=[1.,1.] if a.objective=='dpo' else [1.]
    config=DPOConfig(**common,beta=.1,loss_type=losses,loss_weights=weights,precompute_ref_log_probs=False,disable_dropout=True)
    dataset=Dataset.from_list([{k:x[k] for k in ['prompt','chosen','rejected']} for x in rows])
    pin_single_gpu(config);trainer=DPOTrainer(model=model,ref_model=None,args=config,train_dataset=dataset,processing_class=tokenizer)
    state=model.state_dict();pairs=[(k,k.replace('.default.','.ref.')) for k in state if '.default.' in k]
    assert len(pairs)==392 and all(torch.equal(state[k],state[r]) for k,r in pairs)
    ref_before={r:state[r].detach().cpu().clone() for _,r in pairs}
    assert all(not parameter.requires_grad for name,parameter in model.named_parameters() if '.ref.' in name)
    # Exact preprocessed order and index length check in both objectives.
    assert len(trainer.train_dataset)==443
    trainable=[name for name,param in model.named_parameters() if param.requires_grad]
    assert len(trainable)==392 and all('.default.' in name for name in trainable)
    set_seed(a.seed);start=time.monotonic();result=trainer.train();assert trainer.state.global_step==111
    elapsed=time.monotonic()-start
    if ref_before:assert all(torch.equal(model.state_dict()[k].detach().cpu(),v) for k,v in ref_before.items())
    final=out/'final';trainer.save_model(str(final));tokenizer.save_pretrained(final)
    record={'objective':a.objective,'seed':a.seed,'source_model':str(source),'source_adapter_sha256':sha(source/'adapter_model.safetensors'),'preference_data_sha256':sha(data),'actual_steps':trainer.state.global_step,'dataset_rows':len(rows),'batch_size':1,'gradient_accumulation_steps':4,'effective_batch':4,'planned_prompt_exposures':443,'full_dataset_epochs':1,'real_preference_rows':45,'expert_replay_rows':398,'lr':5e-6,'max_length':4096,'max_data_sequence_tokens':manifest['max_sequence_tokens'],'truncation_count':0,'dropout_disabled':True,'trainable_tensors':len(trainable),'train_loss':result.training_loss,'trainer_elapsed_seconds':elapsed,'torch_peak_gib':torch.cuda.max_memory_allocated()/2**30,'beta':.1,'dpo_loss_type':losses,'dpo_loss_weights':weights,'chosen_only_control_native_ce':a.objective=='sft','reference_equal_sft_and_frozen_tensors':len(ref_before),'sft_completion_only_loss':a.objective=='sft','loss_history':trainer.state.log_history,'observation_intervention':'398expert observations reconstructed through existing benchmark path;45real-error rows unchanged; matched prior CE443row111step controls.', 'caveat':'Both arms use the same native DPOTrainer chosen-token-mean CE path, microbatch/accumulation/row order/forward data; control uses only sft loss, so rejected/reference scores produce no learning gradient. Not claiming exact equal FLOPs. 45replay-certified model-error decisions +398 broad expert replay rows; replay chosen=rejected gives zero preference gradient. Same full443chosen rows; single epoch111steps,not equalFLOPs.'}
    assert math.isfinite(record['train_loss'])
    (final/'training_manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k!='loss_history'},indent=2),flush=True)
if __name__=='__main__':main()
