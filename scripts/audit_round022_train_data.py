#!/usr/bin/env python3
"""Independent raw-trace scoring, program and ordered-row provenance audit."""
from pathlib import Path
import collections,datetime,hashlib,json,os,sys
R=Path('artifacts/reproduction/round-022-2026-10-07');D=Path('data/repro_round022_v2');OLD=R.parent/'round-021-2026-10-06'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=lambda p:[json.loads(x) for x in Path(p).read_text().splitlines()]
def main():
    assert not (R/'independent-train-data-audit.json').exists()
    os.environ.update(CUDA_VISIBLE_DEVICES='',STORE_BACKEND='memory');sys.path.insert(0,'AgenticArxiv')
    from benchmark.metrics import extract_metrics,is_strict_success
    from agents.prompt_templates import get_react_prompt,build_visible_setup_context
    from transformers import AutoTokenizer
    source=rows('data/repro_round014/certified_combined.jsonl');experts=source[:398];errors=source[398:];world=rows(D/'world_combined.jsonl');repeat=rows(D/'repeat_combined.jsonl')
    assert len(world)==len(repeat)==1239 and len(errors)==45
    assert repeat==experts*3+errors and world[:398]==experts and world[-45:]==errors
    original={t['id']:t for t in read(R.parent/'round-013-2026-10-05/collection-oracles.json')};split=read('data/splits/v3_81.json')['split'];train=set(split['train']);held=set(split['dev']+split['iid_test']+split['ood_test']);assert len(train)==51 and not train&held
    plan=read(R/'train-data-plan-v2.json');audit=read(R/'train-data-audit.json')
    for m in [plan,audit]:
        for p,h in m['sha256'].items():assert sha(p)==h,p
    desc=experts[0]['prompt'][0]['content'].split('你有以下工具可以使用：\n\n',1)[1].split('\n\n当前任务：',1)[0];total=0;drift=0;programs=0;row_count=0;assets=0
    for i,w in enumerate(['trainA','trainB']):
        new=rows(D/(w+'-experts.jsonl'));assert world[398*(i+1):398*(i+2)]==new
        oracles={t['id']:t for t in read(R/(w+'-oracles.json'))};traces={t['task_id']:t for t in rows(R/(w+'-gold-traces.jsonl'))};assert set(oracles)==set(traces)==set(original)
        bindings={x['task_id']:x for x in read(R/'entity-bindings.json')['bindings'] if x['world']==w};assert len(bindings)==5
        for tid,t in oracles.items():
            old=original[tid];assert t['parent_task_id'] in train and t['parent_task_id'] not in held
            assert t['expected_tools']==old['expected_tools'] and t.get('setup')==old.get('setup')
            expected=__import__('copy').deepcopy(old['expected_tool_args'])
            if tid in bindings:
                binding=bindings[tid]
                for a in expected:
                    if a.get('ref')==binding['original_literal']:a['ref']=binding['new_literal']
                assert t['task']==old['task'].replace(binding['original_literal'],binding['new_literal'])
            else:assert t['task']==old['task']
            assert t['expected_tool_args']==expected
            for k in ['expected_terminal_mode','expected_terminal_reason','expected_termination','max_iterations']:assert t.get(k)==old.get(k)
            trace=traces[tid];m=extract_metrics(t,{'history':trace['history']},'independent_CPU_gold',0,'')
            total+=1;assert is_strict_success(m),(w,tid,m.to_dict())
            for k in ['strict_success','false_finish']:
                if k in trace['metrics']:drift+=int(m.to_dict().get(k)!=trace['metrics'][k])
            assert [h['action'] for h in trace['history'] if h['action']=='FINISH']==['FINISH'];programs+=1
        for old,x in zip(experts,new):
            tid=x['source_task_id'];step=x['trajectory_step'];history=traces[tid]['history'];t=oracles[tid]
            prefix='\n\n'.join('Thought: '+h['thought']+'\nAction: '+h['action']+'\nObservation: '+h['observation'] for h in history[:step]);initial=build_visible_setup_context(t).strip();context='\n\n'.join(part for part in [initial,prefix] if part)
            assert x['prompt']==[{'role':'user','content':get_react_prompt(t['task'],desc,context)}]
            h=history[step];assert x['chosen']==[{'role':'assistant','content':'Thought: '+h['thought']+'\nAction: '+h['action']}]==x['rejected']
            assert x['chosen'][0]['content'].split('\nAction: ')[0]==old['chosen'][0]['content'].split('\nAction: ')[0]
            for key in ['source_task_id','parent_task_id','trajectory_step','pair_type']:assert x[key]==old[key]
            row_count+=1
        for manifest in [read(OLD/(w+'-asset-manifest.json')),read(R/(w+'-reserve-manifest.json'))]:
            for p,h in manifest['sha256'].items():assert sha(p)==h,p;assets+=1
        curation=read(R/(w+'-resource-curation.json'))
        for p in curation['permutations']:assert sorted(p['before'])==sorted(p['after']) and len(p['after'])==(12 if p['aspect']=='CL' else 10)
    assert total==programs==274 and row_count==796 and drift==0
    assert any(x['source_task_id']=='augp1_state_ref_last_of_10_last12' for x in world)
    tok=AutoTokenizer.from_pretrained('outputs/repro_20261003/sft_e1/final',local_files_only=True);lengths={}
    for label,data in [('world',world),('repeat',repeat)]:
        ls=[]
        for x in data:
            assert x['parent_task_id'] in train and x['parent_task_id'] not in held
            for k in ['chosen','rejected']:
                ids=tok.apply_chat_template(x['prompt']+x[k],tokenize=True)
                if hasattr(ids,'keys'):ids=ids['input_ids']
                ls.append(len(ids))
        assert max(ls)<4096
        lengths[label]={'maximum':max(ls),'mean':sum(ls)/len(ls),'sequences':len(ls),'truncation':0};assert lengths[label]==audit['token_lengths'][label]
    result={'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'raw_gold_traces_independently_rescored':274,'scoring_drift':drift,'new_expert_rows_reconstructed':796,'programs_and_numeric_null_args_preserved':274,'only_title_literal_retargets_each_world':5,'original_thoughts_and_tools_description_preserved':True,'train_parents':51,'heldout_parent_overlap':0,'ordered_rows_each':1239,'unchanged_original_experts_each':398,'unchanged_error_rows_each':45,'repeat_control_exact_3_copies':True,'ref12_training_variant_retained':True,'actual_asset_hashes_checked':assets,'token_lengths':lengths,'same_row_step_budget_not_exact_token_budget':True,'GPU_jobs':0,'new_training_steps':0,'learning_permitted':False,'data_sha256':{str(D/(label+'_combined.jsonl')):sha(D/(label+'_combined.jsonl')) for label in ['world','repeat']}}
    (R/'independent-train-data-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
