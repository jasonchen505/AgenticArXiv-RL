#!/usr/bin/env python3
"""Independent provenance, semantic deduplication and replay-loss checks before training."""
from pathlib import Path
import copy,collections,hashlib,json,os,sys
R=Path('artifacts/reproduction/round-014-2026-10-05');OLD=R.parent/'round-013-2026-10-05';DATA=Path('data/repro_round014')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not (DATA/'certified_combined.jsonl').exists()
    sys.path.insert(0,'AgenticArxiv')
    from rl import trl_compat
    from rl.grpo_reward import parse_react_action,_thought_of
    from transformers import AutoTokenizer,GPT2Config,GPT2LMHeadModel,set_seed
    from datasets import Dataset
    from trl import DPOConfig,DPOTrainer
    import torch
    before=read(R/'training-data-manifest.json')
    for f,h in before['sha256'].items():assert sha(f)==h
    pairs=[json.loads(s) for s in (DATA/'real_preferences.jsonl').read_text().splitlines()]
    replay=[json.loads(s) for s in (DATA/'expert_replay.jsonl').read_text().splitlines()]
    rows=[json.loads(s) for s in (OLD/'rollout_traces.jsonl').read_text().splitlines()]
    audits={x['row_index']:x for x in read(R/'negative-admission-audit.json')['raw_decision_audits'] if x['admitted']}
    seen={};kept=[];removed=[]
    for pair in pairs:
        row=rows[pair['source_row_index']];a=audits[pair['source_row_index']];i=pair['trajectory_step'];hist=row['trajectory']['history'];raw=row['raw_assistant_turns'][i]
        assert raw==a['raw_rejected'] and a['chosen_suffix_strict_success']
        rejected=pair['rejected'][0]['content'];assert rejected in [raw,'Thought:'+raw]
        assert parse_react_action(rejected)==parse_react_action(raw)
        prefix=hist[:i];kind,action=parse_react_action(raw)
        if kind=='call' and any(parse_react_action('Action: '+h['action'])==('call',action) for h in prefix):
            removed.append({'source_row_index':pair['source_row_index'],'reason':'redundant_previously_valid_tool_call'});continue
        # Terminal wording variants do not create independent decisions in a normal task.
        rejected_identity=action if kind=='call' else 'FINISH' if kind=='finish' else raw
        chosen_identity=parse_react_action(pair['chosen'][0]['content'])
        identity=digest([pair['source_task_id'],[(h['action'],h['model_observation']) for h in prefix],chosen_identity,rejected_identity])
        if identity in seen:
            removed.append({'source_row_index':pair['source_row_index'],'reason':'semantic_decision_duplicate','duplicate_of':seen[identity]});continue
        seen[identity]=pair['source_row_index'];pair['semantic_decision_sha256']=identity;kept.append(pair)
    for x in replay:assert x['chosen']==x['rejected'] and x['pair_type']=='expert_replay'
    support={'unique_real_error_decisions':len(kept),'error_semantic_tasks':len({x['source_task_id'] for x in kept}),'error_parent_tasks':len({x['parent_task_id'] for x in kept})}
    g=read(R/'negative-admission-plan.json')['support_gate']
    assert support['unique_real_error_decisions']>=g['minimum_unique_real_error_decisions'] and support['error_semantic_tasks']>=g['minimum_error_semantic_tasks'] and support['error_parent_tasks']>=g['minimum_error_parent_tasks']
    assert len(replay)==398 and len({x['parent_task_id'] for x in replay})==51
    tokenizer=AutoTokenizer.from_pretrained('outputs/repro_20261003/sft_e1/final',local_files_only=True);tokenizer.padding_side='right'
    combined=replay+kept;lengths=[]
    for x in combined:
        for k in ['chosen','rejected']:
            encoded=tokenizer.apply_chat_template(x['prompt']+x[k],tokenize=True)
            if hasattr(encoded,'keys') and 'input_ids' in encoded:encoded=encoded['input_ids']
            lengths.append(len(encoded))
    assert max(lengths)<4096
    # Exercise installed TRL on a real tiny random CPU model, with duplicated replay completions.
    set_seed(123)
    model=GPT2LMHeadModel(GPT2Config(vocab_size=len(tokenizer),n_positions=128,n_embd=8,n_layer=1,n_head=1,resid_pdrop=0.,embd_pdrop=0.,attn_pdrop=0.)).double()
    config=DPOConfig(output_dir='/tmp/agenticarxiv-round014-replay-check',use_cpu=True,bf16=False,fp16=False,report_to='none',per_device_train_batch_size=1,max_length=128,loss_type=['sigmoid','sft'],loss_weights=[1.,1.],beta=.1,disable_dropout=True)
    data=Dataset.from_list([{'prompt':'A paper task: ','chosen':'call the tool','rejected':'call the tool'}])
    trainer=DPOTrainer(model=model,ref_model=copy.deepcopy(model),args=config,train_dataset=data,processing_class=tokenizer)
    batch=next(iter(trainer.get_train_dataloader()))
    trainer.loss_types=['sigmoid'];trainer.loss_weights=[1.]
    pure=trainer._compute_loss(model,batch,return_outputs=False)
    gradient=torch.autograd.grad(pure,model.transformer.wte.weight,retain_graph=True)[0]
    torch.testing.assert_close(pure,torch.tensor(2.,dtype=torch.float64).log())
    assert gradient.abs().max().item()<1e-12
    trainer.loss_types=['sigmoid','sft'];trainer.loss_weights=[1.,1.]
    anchored=trainer._compute_loss(model,batch,return_outputs=False)
    ids=batch['input_ids'];mask=batch['completion_mask'][...,1:].chunk(2)[0].bool()
    logits=model(input_ids=ids,attention_mask=batch['attention_mask'],use_cache=False).logits[...,:-1,:].chunk(2)[0]
    labels=ids[...,1:].chunk(2)[0];ce=torch.nn.functional.cross_entropy(logits[mask],labels[mask])
    torch.testing.assert_close(anchored,pure+ce)
    ga=torch.autograd.grad(anchored,model.transformer.wte.weight,retain_graph=True)[0];gc=torch.autograd.grad(ce,model.transformer.wte.weight)[0]
    torch.testing.assert_close(ga,gc,rtol=1e-4,atol=1e-6)
    loss_check={'status':'passed','hardware':'CPU FP64 tiny random model only','identical_replay_sigmoid_constant':pure.item(),'identical_replay_preference_gradient_max':gradient.abs().max().item(),'anchored_replay_equals_chosen_ce_plus_constant':True,'chosen_ce_gradient_max_difference':(ga-gc).abs().max().item(),'scope':'Replay rows produce no preference signal; rejected equals chosen does not label expert answers as errors. Actual reference/frozen adapters checked during training.'}
    write(R/'replay-loss-integration-check.json',loss_check)
    for name,items in [('certified_preferences.jsonl',kept),('certified_combined.jsonl',combined)]:
        (DATA/name).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in items))
    (DATA/'certified_chosen_control.jsonl').write_text(''.join(json.dumps({'prompt':x['prompt'],'completion':x['chosen'],'pair_type':x['pair_type']},ensure_ascii=False)+'\n' for x in combined))
    write(R/'independent-data-check.json',{'input_candidate_pairs':len(pairs),'support':support,'removed':removed,'source_rejected_provenance_verified':True,'all_previous_chosen_certifications_retained':True,'terminal_wording_not_independent_support':True,'replay_loss_check':loss_check,'support_gate_passed':True})
    manifest={**before,**support,'real_preference_rows':len(kept),'combined_rows':len(combined),'max_sequence_tokens':max(lengths),'semantic_dedup_completed':True,'support_gate_passed':True,'sha256':{str(p):sha(p) for p in DATA.glob('certified*.jsonl')},'negative_type_counts':dict(collections.Counter(x['negative_type'] for x in kept))}
    write(R/'certified-training-data-manifest.json',manifest);print(json.dumps(manifest,indent=2),flush=True)
if __name__=='__main__':main()
