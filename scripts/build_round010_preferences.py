#!/usr/bin/env python3
"""Train-only step preference dataset with matched chosen-only SFT control."""
from pathlib import Path
import collections,hashlib,json,os,sys
R=Path('artifacts/reproduction/round-010-2026-10-04');R.mkdir(parents=True,exist_ok=True)
DATA=Path('data/repro_round010/preferences.jsonl')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def main():
    assert not DATA.exists()
    sys.path[:0]=['scripts','AgenticArxiv']
    from benchmark.tasks_expanded import get_expanded_tasks
    from generate_parametric_sft_data import build_parametric_tasks
    from benchmark.metrics import classify_blocked_terminal_semantics
    split=json.loads(Path('data/splits/v3_81.json').read_text())['split'];train=set(split['train']);held=set(sum((split[s] for s in ['dev','iid_test','ood_test']),[]))
    specs=[t for t in get_expanded_tasks() if t['id'] in train]+[d.spec.to_task() for d in build_parametric_tasks(include_t5=True)]
    by={t['id']:t for t in specs}
    source=Path('data/repro_20261002/sft_train.jsonl')
    allrows=[json.loads(s) for s in source.read_text().splitlines()]
    errors={'invalid_reference':'请求中的 ref 序号无效或超出 1-based 索引范围，无法执行',
            'paper_not_found':'指定的 arXiv 论文 ID 不在快照记录中，无法下载该论文',
            'missing_context':'当前会话缺少必要上下文，无法解析用户所指的论文'}
    preferred=[];counts=collections.Counter();seen=set();parents=set()
    for row in allrows:
        t=by[row['source_task_id']];blocked=t.get('expected_terminal_mode')=='blocked'
        # Six existing linguistic wrappers for blocked states, one for normal
        # states; no new semantic instances or held-out examples introduced.
        if row['augmentation']['thought_variant']!=0:continue
        if not blocked and row['augmentation']['task_variant']!=0:continue
        assert row['parent_task_id'] in train and row['parent_task_id'] not in held
        prompt,chosen=[m['content'] for m in row['messages']]
        idx=row['augmentation']['task_variant'] if blocked else row['trajectory_step']
        if blocked:
            correct=t['expected_terminal_reason']
            options=[v for k,v in errors.items() if k!=correct]+['所有步骤均已成功执行，任务已完成']
            rejected_thought=options[idx%3]
            chosen_thought=chosen.split('\nAction:',1)[0].removeprefix('Thought:').strip()
            assert classify_blocked_terminal_semantics(t,[{'thought':chosen_thought,'action':'FINISH'}])=='explained_block'
            assert classify_blocked_terminal_semantics(t,[{'thought':rejected_thought,'action':'FINISH'}])!='explained_block'
            kind='blocked_wrong_reason' if idx%3<2 else 'blocked_false_completion'
        else:
            rejected_thought=list(errors.values())[idx%3];kind='feasible_wrong_refusal'
        rejected='Thought: '+rejected_thought+'\nAction: FINISH'
        assert chosen!=rejected and prompt==row['messages'][0]['content']
        identity=hashlib.sha256(json.dumps([prompt,chosen,rejected],ensure_ascii=False).encode()).hexdigest()
        assert identity not in seen;seen.add(identity);parents.add(row['parent_task_id']);counts[kind]+=1
        preferred.append({'prompt':[{'role':'user','content':prompt}],'chosen':[{'role':'assistant','content':chosen}],'rejected':[{'role':'assistant','content':rejected}], 'source_task_id':row['source_task_id'],'parent_task_id':row['parent_task_id'],'trajectory_step':row['trajectory_step'],'negative_type':kind,'pair_sha256':identity,'source_sample_sha256':row['sample_sha256']})
    assert len(preferred)==448 and len(parents)==51
    # Tokenize before any policy load: verify both chosen/rejected sequences.
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained('outputs/repro_20261003/sft_e1/final',local_files_only=True)
    lengths=[len(tok.apply_chat_template(x['prompt']+x[k],tokenize=True)) for x in preferred for k in ['chosen','rejected']]
    assert max(lengths)<4096
    DATA.parent.mkdir(parents=True,exist_ok=True)
    DATA.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in preferred))
    chosen_file=DATA.with_name('chosen_control.jsonl')
    chosen_file.write_text(''.join(json.dumps({'prompt':x['prompt'],'completion':x['chosen'],'pair_sha256':x['pair_sha256']},ensure_ascii=False)+'\n' for x in preferred))
    manifest={'pairs':len(preferred),'train_parent_coverage':len(parents),'train_semantic_instances':len({x['source_task_id'] for x in preferred}),'heldout_parent_overlap':len(parents&held),'negative_counts':dict(counts),'synthetic_negatives':True,'negative_construction':'Predeclared three train reason templates or false completion. No model rollout or held-out trace read.','blocked_sampling':'All six original task wrappers, thought variant0; feasible only original wrapper0/thought0. Linguistic variants not independent semantic support.','chosen_control':'Same prompts, chosen texts and order, no rejected texts. Extra SFT separates preference effect from continued chosen-only learning.','max_sequence_tokens':max(lengths),'truncated_sequences':0,'max_length':4096,'sha256':{str(p):sha(p) for p in [DATA,chosen_file,source,Path('data/splits/v3_81.json')]}}
    write(R/'preference-data-manifest.json',manifest)
    print(json.dumps(manifest,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
