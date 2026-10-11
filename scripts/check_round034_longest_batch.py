import os,sys
sys.path[:0]=['AgenticArxiv','scripts']
from round034_core import R,D,PY,read,write,verify
from train_round034_plan import admission,construct,collate
def main():
 import torch
 gpu=admission();p=read(R/'preflight-plan.json');verify(p);trainer,tok,model=construct(p,'mixed49',D/'longest-preflight');model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});assert model.is_gradient_checkpointing;model.train();trainer.current_gradient_accumulation_steps=4;before={n:v.detach().cpu().clone() for n,v in model.named_parameters() if v.requires_grad};idx=max(range(827),key=lambda i:len(trainer.train_dataset[i]['input_ids']));x=trainer._prepare_inputs(collate([trainer.train_dataset[idx]]))
 with trainer.compute_loss_context_manager():loss=trainer.compute_loss(model,x)
 trainer.accelerator.backward(loss/4);assert any(v.grad is not None and torch.isfinite(v.grad).all() and bool(v.grad.abs().sum()>0) for v in model.parameters() if v.requires_grad);assert all(torch.equal(model.get_parameter(n).detach().cpu(),v) for n,v in before.items());write(R/'checkpointed-preflight-result.json',{'passed':True,'physical_gpu':gpu,'optimizer_steps':0,'real_forward_backward_calls':1,'source_trainable_tensors_unchanged':392,'gradient_checkpointing_enabled_same_as_training':True,'longest_full_tokens':len(trainer.train_dataset[idx]['input_ids']),'row_index':trainer.train_dataset[idx]['row_index'],'loss':float(loss.detach()),'finite_nonzero_adapter_gradient':True,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30})
if __name__=='__main__':main()
