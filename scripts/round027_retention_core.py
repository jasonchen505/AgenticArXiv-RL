"""Completion forward KL; native CE backward followed by equivalent anchor backward."""
import json,random
from pathlib import Path
import torch
from trl import DPOTrainer
from trl.trainer.dpo_trainer import use_adapter,disable_gradient_checkpointing

def encode_anchor(tokenizer,row):
 prompt=tokenizer.apply_chat_template(row['prompt'],add_generation_prompt=True,tokenize=True,return_dict=False)
 full=tokenizer.apply_chat_template(row['prompt']+row['chosen'],tokenize=True,return_dict=False)
 assert full[:len(prompt)]==prompt and 0<len(prompt)<len(full)<4096
 return {'input_ids':torch.tensor([full]),'attention_mask':torch.ones(1,len(full),dtype=torch.long),'completion_mask':torch.tensor([[0]*len(prompt)+[1]*(len(full)-len(prompt))])}

def forward_kl(policy_logits,teacher_logits,mask):
 """Full vocabulary KL(teacher||policy), mean over masked next-token positions."""
 assert policy_logits.shape==teacher_logits.shape and mask.shape==policy_logits.shape[:-1] and mask.any()
 teacher_logp=torch.log_softmax(teacher_logits.detach()[mask.bool()].float(),dim=-1)
 policy_logp=torch.log_softmax(policy_logits[mask.bool()].float(),dim=-1)
 return (teacher_logp.exp()*(teacher_logp-policy_logp)).sum(-1).mean()

def anchor_order(seed,n=443,exposures=795):
 order=list(range(n));random.Random(seed).shuffle(order)
 return [order[i%n] for i in range(exposures)]

class RetentionTrainer(DPOTrainer):
 def __init__(self,*args,anchors,anchor_seed,retention_weight=.1,**kwargs):
  super().__init__(*args,**kwargs)
  self.anchors=anchors;self.anchor_order=anchor_order(anchor_seed);self.anchor_calls=0;self.retention_weight=retention_weight;self.anchor_logs=[]
  assert self.loss_types==['sft'] and self.loss_weights==[1.] and self.accelerator.gradient_accumulation_steps==1
  assert self.compute_loss_func is None and not self.use_apex

 def anchor_loss(self,model,index):
  inputs={k:v.to(self.accelerator.device) for k,v in self.anchors[index].items()};mask=(inputs['completion_mask'][:,1:]*inputs['attention_mask'][:,1:]).bool();kwargs={k:inputs[k] for k in ['input_ids','attention_mask']};kwargs['use_cache']=False
  raw=self.accelerator.unwrap_model(model)
  with torch.no_grad(),disable_gradient_checkpointing(raw,self.args.gradient_checkpointing_kwargs),use_adapter(raw,'ref'):
   teacher=raw(**kwargs).logits[:,:-1][mask].detach()
  assert raw.active_adapter=='default' and all(not p.requires_grad for n,p in raw.named_parameters() if '.ref.' in n)
  policy=model(**kwargs).logits[:,:-1][mask]
  # Masked gather above aligns each completion token with its previous position.
  loss=forward_kl(policy,teacher,torch.ones(policy.shape[:-1],device=policy.device,dtype=torch.bool))
  assert torch.isfinite(loss) and loss.item()>-1e-5
  return loss,int(mask.sum())

 def training_step(self,model,inputs,num_items_in_batch=None):
  # Native preference batches contain no labels; Trainer's token-count scaling is inactive.
  assert num_items_in_batch is None and 'labels' not in inputs
  # Free native CE activations before the additional anchor pass; gradients add before optimizer/clip.
  ce=super().training_step(model,inputs,num_items_in_batch)
  index=self.anchor_order[self.anchor_calls];self.anchor_calls+=1
  with self.compute_loss_context_manager():kl,tokens=self.anchor_loss(model,index)
  scale=self.current_gradient_accumulation_steps
  self.accelerator.backward(self.retention_weight*kl/scale)
  self.anchor_logs.append({'microbatch':self.anchor_calls-1,'optimizer_step_before':self.state.global_step,'anchor_index':index,'completion_tokens':tokens,'forward_kl':kl.detach().item(),'accumulation_group_size':scale})
  self._metrics['train']['retention/forward_kl'].append(kl.detach().item())
  return ce+self.retention_weight*kl.detach()/scale
