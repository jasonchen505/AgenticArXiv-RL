import os,sys,json,math,time,subprocess,collections
from pathlib import Path
sys.path[:0]=['AgenticArxiv','scripts']
from rl import trl_compat
import torch
import torch.nn.functional as F
from torch.utils.data import SequentialSampler
from transformers import Trainer,TrainingArguments,AutoTokenizer,set_seed
from trl.trainer.utils import disable_dropout_in_model
from rl.train_grpo import _load_policy
from rl.precision import pin_single_gpu
from round036_core import R,D,read,rows,sha,write,verify,encode

class CompletionTrainer(Trainer):
 def __init__(self,*a,**kw):super().__init__(*a,**kw);self.model_accepts_loss_kwargs=False;self.exposures=[]
 def _get_train_sampler(self,train_dataset=None):return SequentialSampler(train_dataset if train_dataset is not None else self.train_dataset)
 def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
  indices=inputs.pop('row_index');labels=inputs.pop('labels');out=model(**inputs,use_cache=False);mask=labels[:,1:]!=-100;assert mask.any();loss=F.cross_entropy(out.logits[:,:-1][mask].float(),labels[:,1:][mask]);assert torch.isfinite(loss)
  if model.training:self.exposures.extend([{'row_index':int(i),'optimizer_step_before':self.state.global_step,'completion_tokens':int(mask.sum()),'raw_mean_completion_loss':float(loss.detach()),'accumulation_group_size':self.current_gradient_accumulation_steps,'learning_rate_before':self.optimizer.param_groups[0]['lr'] if self.optimizer else None} for i in indices])
  return (loss,out) if return_outputs else loss
def collate(batch):
 assert len(batch)==1;x=batch[0];return {k:torch.tensor([v],dtype=torch.long) for k,v in x.items()}
def admission():
 gpu=os.environ.get('CUDA_VISIBLE_DEVICES');assert gpu in ['4','7'];r=[x.strip() for x in subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')];processes=subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True);assert int(r[1])>=23000 and not any(x.split(',')[0].strip()==r[0] for x in processes.splitlines());return int(gpu)
def construct(p,arm,out):
 seed=49;set_seed(seed);tok=AutoTokenizer.from_pretrained(p['source_model'],local_files_only=True);tok.padding_side='right';data=rows(p['training_data']);encoded=[dict(encode(tok,data[i]),row_index=i) for i in p['training_order']];model=_load_policy(p['source_model'],qlora=True,base_model=None,lora_r=16,lora_alpha=32,lora_dropout=.05);disable_dropout_in_model(model)
 args=TrainingArguments(output_dir=str(out),num_train_epochs=1.,per_device_train_batch_size=1,gradient_accumulation_steps=4,learning_rate=5e-6 if arm=='plan49' else 0.,bf16=True,fp16=False,seed=seed,data_seed=seed,logging_steps=1,logging_first_step=True,save_strategy='no',eval_strategy='no',report_to='none',optim='paged_adamw_8bit',gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},lr_scheduler_type='cosine',warmup_ratio=.03,weight_decay=0.,max_grad_norm=1.,remove_unused_columns=False,dataloader_num_workers=0);pin_single_gpu(args);trainer=CompletionTrainer(model=model,args=args,train_dataset=encoded,data_collator=collate,processing_class=tok);assert trainer.accelerator.gradient_accumulation_steps==1 and args.n_gpu==1
 return trainer,tok,model
def main():
 import argparse;a=argparse.ArgumentParser();a.add_argument('--arm',choices=['plan49','lr0_49'],required=True);a.add_argument('--preflight',action='store_true');args=a.parse_args();gpu=admission();p=read(R/'experiment-plan.json' if not args.preflight else R/'preflight-plan.json');verify(p);out=Path('outputs/repro_round036')/args.arm;assert args.preflight or not out.exists();trainer,tok,model=construct(p,args.arm,out)
 trainables={n:v.detach().cpu().clone() for n,v in model.named_parameters() if v.requires_grad};assert len(trainables)==392
 if args.preflight:
  model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});assert model.is_gradient_checkpointing;trainer.current_gradient_accumulation_steps=4;trainer.model.train();idx=max(range(512),key=lambda i:len(trainer.train_dataset[i]['input_ids']));inputs=collate([trainer.train_dataset[idx]]);inputs=trainer._prepare_inputs(inputs)
  with trainer.compute_loss_context_manager():loss=trainer.compute_loss(model,inputs)
  trainer.accelerator.backward(loss/4);assert any(v.grad is not None and torch.isfinite(v.grad).all() and bool(v.grad.abs().sum()>0) for v in model.parameters() if v.requires_grad);assert all(torch.equal(model.get_parameter(n).detach().cpu(),v) for n,v in trainables.items());write(R/'preflight-result.json',{'passed':True,'physical_gpu':gpu,'real_forward_backward_calls':1,'optimizer_steps':0,'source_trainable_tensors_unchanged':392,'finite_nonzero_adapter_gradient':True,'loss':float(loss.detach()),'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'completion_only_objective':True,'longest_full_tokens':len(trainer.train_dataset[idx]['input_ids']),'gradient_checkpointing_enabled_same_as_training':True,'training_rows':512,'source_model_unchanged':True});return
 assert p['learning_permitted'];out.mkdir(parents=True,exist_ok=True);started=time.monotonic();result=trainer.train();assert trainer.state.global_step==128 and len(trainer.exposures)==512 and [x['row_index'] for x in trainer.exposures]==p['training_order'];assert all(x['accumulation_group_size']==4 for x in trainer.exposures);changed=sum(not torch.equal(model.get_parameter(n).detach().cpu(),v) for n,v in trainables.items());assert changed== (392 if args.arm=='plan49' else 0)
 nonzero_steps=len({x['optimizer_step_before'] for x in trainer.exposures if x['learning_rate_before']>0});assert nonzero_steps==(127 if args.arm=='plan49' else 0);final=out/'final';trainer.save_model(str(final));tok.save_pretrained(final);assert math.isfinite(result.training_loss);write(final/'training_manifest.json',{'arm':args.arm,'physical_gpu':gpu,'seed':49,'dataset_rows':512,'plan_unique128_exposures512':True,'native_training_rows':0,'actual_optimizer_steps':128,'positive_lr_learning_steps':nonzero_steps,'microbatch_calls':512,'source_model':p['source_model'],'source_adapter_sha256':sha(Path(p['source_model'])/'adapter_model.safetensors'),'changed_trainable_tensors':changed,'training_data_sha256':sha(p['training_data']),'completion_only_mean_CE_no_KL_or_RL':True,'preflight_not_learning':True,'dropout_disabled':True,'order':p['training_order'],'exposures':trainer.exposures,'log_history':trainer.state.log_history,'train_loss':result.training_loss,'elapsed_seconds':time.monotonic()-started,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30});print({'arm':args.arm,'steps':128,'nonzero_lr_steps':nonzero_steps,'changed_tensors':changed},flush=True)
if __name__=='__main__':main()
