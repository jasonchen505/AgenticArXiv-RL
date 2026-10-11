#!/usr/bin/env python3
"""Math, masks, frozen source teacher and worst-length GPU backward; zero optimizer steps."""
from pathlib import Path
import datetime,hashlib,json,math,os,sys,tempfile
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1]/'AgenticArxiv')]
from rl import trl_compat
import torch
from round027_retention_core import forward_kl,anchor_order
from train_round027_retention import construct,admission
R=Path('artifacts/reproduction/round-027-2026-10-08');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 R.mkdir(exist_ok=True);assert not (R/'preflight-result.json').exists() and not Path('outputs/repro_round027').exists()
 src=Path('outputs/repro_20261003/sft_e1/final');paths=[Path(__file__),Path('scripts/round027_retention_core.py'),Path('scripts/train_round027_retention.py'),Path('data/repro_round026/source-train-anchors.jsonl'),Path('data/repro_round024/curriculum_combined.jsonl'),Path('data/repro_round024/matched_repeat_combined.jsonl'),src/'adapter_model.safetensors',Path('artifacts/reproduction/round-026-2026-10-08/retention-independent-audit.json')]
 plan={'registered_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'only_preflight_no_learning':True,'zero_optimizer_steps':True,'temporary_in_memory_perturbation_for_gradient_probe_then_exact_reset':True,'no_candidate_checkpoint_or_evaluation':'Worst length chosen CE/backward + anchorKL/backward, not a learning pilot.','fixed_retention_weight':.1,'sha256':{str(p):sha(p) for p in paths}};write(R/'preflight-plan-v2.json',plan)
 # Analytic direction, masking, averaging, detached teacher, and split-backward accumulation.
 teacher=torch.log(torch.tensor([[[.8,.2],[.1,.9],[.6,.4]]],dtype=torch.float64)).requires_grad_();policy=torch.log(torch.tensor([[[.5,.5],[.9,.1],[.5,.5]]],dtype=torch.float64)).requires_grad_();mask=torch.tensor([[True,False,True]])
 kl=forward_kl(policy,teacher,mask);expected=(.8*math.log(1.6)+.2*math.log(.4)+.6*math.log(1.2)+.4*math.log(.8))/2;assert abs(kl.item()-expected)<1e-6
 kl.backward();assert teacher.grad is None and torch.equal(policy.grad[0,1],torch.zeros(2,dtype=policy.dtype)) and policy.grad.abs().sum()>0
 equal=forward_kl(teacher.detach(),teacher.detach(),mask);assert equal.item()==0
 for accumulation in [3,4]:
  x=torch.tensor([.3,-.7],requires_grad=True);ce=(x**2).sum();logits=x.reshape(1,2);t=torch.log(torch.tensor([[.8,.2]]));l=forward_kl(logits,t,torch.tensor([True]));((ce+.1*l)/accumulation).backward();joint=x.grad.clone()
  y=x.detach().clone().requires_grad_();(y.square().sum()/accumulation).backward();(.1*forward_kl(y.reshape(1,2),t,torch.tensor([True]))/accumulation).backward();assert torch.allclose(joint,y.grad,atol=1e-7,rtol=1e-7)
 gpu=admission()
 with tempfile.TemporaryDirectory(prefix='retention027-preflight-') as tmp:
  trainer,tok,model,data=construct('curriculum',43,Path(tmp));state=model.state_dict();pairs=[(k,k.replace('.default.','.ref.')) for k in state if '.default.' in k];assert len(pairs)==392 and all(torch.equal(state[k],state[r]) for k,r in pairs)
  before={k:state[k].detach().cpu().clone() for k,_ in pairs};ref={r:state[r].detach().cpu().clone() for _,r in pairs}
  lengths=[len(x['prompt_ids'])+max(len(x['chosen_ids']),len(x['rejected_ids'])) for x in trainer.train_dataset];anchor_lengths=[x['input_ids'].shape[1] for x in trainer.anchors];imax=max(range(443),key=anchor_lengths.__getitem__);sample=trainer.train_dataset[max(range(795),key=lengths.__getitem__)]
  assert max(lengths)<4096 and max(anchor_lengths)<4096
  batch=trainer.data_collator([sample]);trainer.anchor_order=[imax];trainer.current_gradient_accumulation_steps=4;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
  torch.cuda.reset_peak_memory_stats()
  with torch.autocast('cuda',dtype=torch.bfloat16):loss=trainer.training_step(model,batch)
  initial_kl=trainer.anchor_logs[-1]['forward_kl'];assert abs(initial_kl)<1e-5 and torch.isfinite(loss)
  assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if '.default.' in n)
  assert all(p.grad is None and not p.requires_grad for n,p in model.named_parameters() if '.ref.' in n)
  model.zero_grad(set_to_none=True);param=next(p for n,p in model.named_parameters() if '.lora_B.default.' in n)
  saved=param.detach().clone()
  with torch.no_grad():param.add_(.01)
  with torch.autocast('cuda',dtype=torch.bfloat16):nonzero,tokens=trainer.anchor_loss(model,imax)
  assert nonzero.item()>1e-7;nonzero.backward()
  assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if '.default.' in n) and all(p.grad is None for n,p in model.named_parameters() if '.ref.' in n)
  with torch.no_grad():param.copy_(saved)
  model.zero_grad(set_to_none=True)
  assert all(torch.equal(model.state_dict()[k].detach().cpu(),v) for k,v in before.items()) and all(torch.equal(model.state_dict()[r].detach().cpu(),v) for r,v in ref.items())
  for p,h in plan['sha256'].items():assert sha(p)==h,p
  result={'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'physical_gpu':gpu,'optimizer_steps':0,'candidate_checkpoints':0,'model_evaluation_predictions':0,'analytic_forward_KL_direction_verified':True,'completion_prompt_padding_mask_verified':True,'full_vocab_mean_completion_normalization':True,'teacher_detached_no_grad':True,'equal_policy_teacher_KL':initial_kl,'perturbed_policy_KL':nonzero.item(),'nonzero_policy_gradient_verified':True,'accumulation3_and4_split_backward_equals_joint_gradient':True,'worst_native_pair_length':max(lengths),'worst_anchor_length':max(anchor_lengths),'probe_anchor_index':imax,'probe_completion_tokens':tokens,'all392policy_tensors_reset_exactly':True,'all392reference_tensors_equal_source':True,'source_files_unchanged':True,'torch_peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'torch_peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'anchor_order_seed43':anchor_order(43),'anchor_order_seed44':anchor_order(44),'sha256':plan['sha256']}
  write(R/'preflight-result.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['anchor_order_seed43','anchor_order_seed44','sha256']},indent=2))
if __name__=='__main__':main()
