#!/usr/bin/env python3
"""CPU integration check: native combined loss adds chosen-completion CE only."""
import copy,json,os,sys
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','This check must run CPU-only'
sys.path.insert(0,'AgenticArxiv')
from rl import trl_compat
import torch
from datasets import Dataset
from transformers import AutoTokenizer,GPT2Config,GPT2LMHeadModel,set_seed
from trl import DPOConfig,DPOTrainer
set_seed(123)
tokenizer=AutoTokenizer.from_pretrained('outputs/repro_20261003/sft_e1/final',local_files_only=True)
tokenizer.padding_side='right'
model=GPT2LMHeadModel(GPT2Config(vocab_size=len(tokenizer),n_positions=128,n_embd=8,n_layer=1,n_head=1,resid_pdrop=0.,embd_pdrop=0.,attn_pdrop=0.)).double()
config=DPOConfig(output_dir='/tmp/agenticarxiv-round014-control-loss-check',use_cpu=True,bf16=False,fp16=False,report_to='none',per_device_train_batch_size=1,max_length=128,loss_type=['sigmoid'],beta=.1,disable_dropout=True)
data=Dataset.from_list([{'prompt':'A paper task: ','chosen':'call the tool','rejected':'stop immediately'}])
trainer=DPOTrainer(model=model,ref_model=copy.deepcopy(model),args=config,train_dataset=data,processing_class=tokenizer)
batch=next(iter(trainer.get_train_dataloader()))
trainer.loss_types=['sigmoid'];trainer.loss_weights=[1.]
pure=trainer._compute_loss(model,batch,return_outputs=False)
trainer.loss_types=['sft'];trainer.loss_weights=[1.]
ce_only=trainer._compute_loss(model,batch,return_outputs=False)
trainer.loss_types=['sigmoid','sft'];trainer.loss_weights=[1.,1.]
combined=trainer._compute_loss(model,batch,return_outputs=False)
ids=batch['input_ids'];mask=batch['completion_mask'][... ,1:].chunk(2)[0].bool()
logits=model(input_ids=ids,attention_mask=batch['attention_mask'],use_cache=False).logits[..., :-1,:].chunk(2)[0]
labels=ids[..., 1:].chunk(2)[0]
chosen_ce=torch.nn.functional.cross_entropy(logits[mask],labels[mask])
torch.testing.assert_close(ce_only,chosen_ce)
torch.testing.assert_close(combined,pure+ce_only)
torch.testing.assert_close(combined,pure+chosen_ce)
assert mask.sum()>0 and not mask.all()
grad_combined=torch.autograd.grad(combined,model.transformer.wte.weight,retain_graph=True)[0]
grad_expected=torch.autograd.grad(pure+chosen_ce,model.transformer.wte.weight)[0]
torch.testing.assert_close(grad_combined,grad_expected,rtol=1e-4,atol=1e-6)
result={'status':'passed','hardware':'CPU FP64 only; random tiny model, no candidate training','native_combined_loss':combined.item(),'native_chosen_only_loss':ce_only.item(),'control_native_ce_only_verified':True,'native_sigmoid_loss':pure.item(),'independently_computed_chosen_completion_ce':chosen_ce.item(),'loss_additivity':True,'embedding_gradient_additivity':True,'gradient_max_absolute_difference':(grad_combined-grad_expected).abs().max().item(),'chosen_completion_mask_tokens':mask.sum().item(),'prompt_and_rejected_excluded_from_ce':True,'scope':'Native installed-TRL loss composition and completion mask; actual saved SFT reference and optimizer steps checked during real training. Both arms use native DPOTrainer CE path and same microbatch/accumulation weighting. Control-only native CE equals independent token-mean chosen CE.'}
Path('artifacts/reproduction/round-014-2026-10-05/control-loss-integration-check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
