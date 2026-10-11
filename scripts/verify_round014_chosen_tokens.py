#!/usr/bin/env python3
"""Verify DPO and SFT preprocess identical chosen tokens/masks on all actual rows."""
from pathlib import Path
import copy,json,os,sys,hashlib
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path.insert(0,'AgenticArxiv')
from rl import trl_compat
import torch
from datasets import Dataset
from transformers import AutoTokenizer,GPT2Config,GPT2LMHeadModel
from trl import DPOConfig,DPOTrainer,SFTConfig,SFTTrainer
rows=[json.loads(s) for s in Path('data/repro_round014/certified_combined.jsonl').read_text().splitlines()]
tok=AutoTokenizer.from_pretrained('outputs/repro_20261003/sft_e1/final',local_files_only=True);tok.padding_side='right'
model=GPT2LMHeadModel(GPT2Config(vocab_size=len(tok),n_positions=4096,n_embd=8,n_layer=1,n_head=1))
common=dict(use_cpu=True,bf16=False,fp16=False,report_to='none',max_length=4096,per_device_train_batch_size=1)
dpo=DPOTrainer(model=model,ref_model=copy.deepcopy(model),args=DPOConfig(output_dir='/tmp/round014-token-dpo',**common,loss_type=['sigmoid','sft'],loss_weights=[1.,1.],beta=.1),train_dataset=Dataset.from_list([{k:x[k] for k in ['prompt','chosen','rejected']} for x in rows]),processing_class=tok)
sft=SFTTrainer(model=copy.deepcopy(model),args=SFTConfig(output_dir='/tmp/round014-token-sft',**common,completion_only_loss=True,packing=False),train_dataset=Dataset.from_list([{'prompt':x['prompt'],'completion':x['chosen']} for x in rows]),processing_class=tok)
assert len(dpo.train_dataset)==len(sft.train_dataset)==443
counts=[];signatures=[]
for i in range(443):
    db=dpo.data_collator([dpo.train_dataset[i]]);sb=sft.data_collator([sft.train_dataset[i]])
    di=db['input_ids'][0][db['attention_mask'][0].bool()];si=sb['input_ids'][0][sb['attention_mask'][0].bool()]
    dm=db['completion_mask'][0][db['attention_mask'][0].bool()].bool();sm=(sb['labels'][0][sb['attention_mask'][0].bool()]!=-100)
    assert torch.equal(di,si),(i,'chosen_input_ids')
    assert torch.equal(dm,sm),(i,'chosen_completion_mask')
    assert sm.sum()>0
    counts.append(int(sm[1:].sum()));signatures.append(di.tolist())
record={'status':'passed','hardware':'CPU; actual443trainingrows; random tiny model used only for trainer preprocessing, no forward or training','chosen_input_token_sequences_equal':443,'chosen_completion_masks_equal':443,'chosen_shifted_supervised_tokens':sum(counts),'minimum_shifted_chosen_tokens':min(counts),'chosen_token_sequences_sha256':hashlib.sha256(json.dumps(signatures).encode()).hexdigest(),'chosen_ce_same_token_support':True,'equal_FLOPs_claim':False}
Path('artifacts/reproduction/round-014-2026-10-05/chosen-token-integration-check.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
