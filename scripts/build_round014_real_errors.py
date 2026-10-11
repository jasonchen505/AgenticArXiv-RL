#!/usr/bin/env python3
"""Replay-certified real rejected decisions; no heldout data or negative templates."""
from pathlib import Path
import collections,hashlib,inspect,json,os,sys,tempfile
R=Path('artifacts/reproduction/round-014-2026-10-05');OLD=R.parent/'round-013-2026-10-05';DATA=Path('data/repro_round014')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
    assert not (DATA/'real_preferences.jsonl').exists()
    gate=read(R/'negative-admission-plan.json')
    for f,h in gate['source_sha256'].items():assert sha(OLD/f)==h,f
    sys.path[:0]=['scripts','AgenticArxiv'];os.environ['STORE_BACKEND']='memory'
    with tempfile.TemporaryDirectory(prefix='round014-cert-') as temp:
        for key in ['PDF_RAW_PATH','PDF_TRANSLATED_PATH','PDF_FIGURES_PATH']:os.environ[key]=str(Path(temp)/key.lower())
        from tools.bootstrap import require_all_tools
        require_all_tools('Round014 same-state real-error certification')
        from rl.multiturn_env import AgenticArxivMultiTurnEnv
        from rl.grpo_reward import _prepare_rollout_environment,_dispatch_environment_tool,parse_react_action,_thought_of,build_prompt_dataset
        from benchmark.metrics import extract_metrics,is_strict_success,_match_arg_value,_same_paper
        from benchmark.task_spec import reference_terminal_thought
        from agents.prompt_templates import build_visible_setup_context,format_tool_description,get_react_prompt
        from tools.tool_registry import registry
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained('outputs/repro_20261003/sft_e1/final',local_files_only=True)
        tools_desc=format_tool_description(registry.list_tools())
        tasks={t['id']:t for t in read(OLD/'collection-oracles.json')}
        raw=[json.loads(s) for s in (OLD/'rollout_traces.jsonl').read_text().splitlines()]
        inventory=read(OLD/'failure-inventory.json');decisions=[];pairs=[];seen={};exclude=collections.Counter()
        split=read('data/splits/v3_81.json')['split'];train=set(split['train']);held=set(split['dev']+split['iid_test']+split['ood_test'])
        for record in inventory['failures']:
            x=raw[record['row_index']];t=tasks[x['task_id']];hist=x['trajectory']['history'];turns=x['raw_assistant_turns']
            assert t['parent_task_id'] in train and t['parent_task_id'] not in held
            audit={'row_index':record['row_index'],'task_id':t['id'],'parent_task_id':t['parent_task_id'],'admitted':False}
            decisions.append(audit)
            def omit(reason):audit['exclusion']=reason;exclude[reason]+=1
            if x['trajectory'].get('clipped') or len(turns)!=len(hist):omit('clipped_or_turn_alignment');continue
            env=AgenticArxivMultiTurnEnv('data/repro_20261002/snapshot_curated_v1.json')
            setups=_prepare_rollout_environment(env,t['id'],tasks);assert setups==x['trajectory']['setup_actions']
            prefix=[];idx=None;kind=None;chosen_action=None;error=None
            expected=list(zip(t['expected_tools'],t['expected_tool_args']))
            for i,(step,text) in enumerate(zip(hist,turns)):
                # A hallucinated observation or extra action marker is not silently removed.
                parsed_kind,action=parse_react_action(text)
                normalized=('FINISH' if parsed_kind=='finish' else 'PARSE_ERROR' if parsed_kind=='parse_error' else json.dumps(action,ensure_ascii=False))
                if normalized!=step['action'] or _thought_of(text)!=step['thought'] or 'Observation:' in text or text.count('Action:')>1:error='raw_executed_turn_mismatch';break
                if parsed_kind=='parse_error':idx=i;kind='malformed_action';break
                if t.get('expected_terminal_mode')=='blocked':
                    idx=i
                    if parsed_kind=='call':
                        try:_dispatch_environment_tool(env,action['name'],action['args'])
                        except Exception as exc:
                            if '快照缺失' in str(exc):error='missing_snapshot_artifact'
                            else:kind='blocked_invalid_call';audit['rejected_execution_error']=str(exc)
                        else:error='ambiguous_blocked_action'
                    elif any(w in step['thought'] for w in ['任务已完成','操作已经完成','所有步骤均已成功','整个任务要求的操作已经完成']):kind='blocked_false_completion'
                    else:error='ambiguous_or_reason_only_blocked_text'
                    break
                if i>=len(expected):error='possibly_harmless_extra_action';break
                name,args=expected[i]
                chosen_action={'name':name,'args':args}
                if parsed_kind=='finish':idx=i;kind='premature_finish';break
                if action['name']!=name:
                    # Exclude possible reordering/extra diagnostics, rather than label them wrong by position alone.
                    if any(action['name']==n for n,a in expected[i+1:]):error='possible_order_alternative';break
                    if action['name']=='get_paper_cache_status' and not any(w in t['task'] for w in ['只','仅','不要']):error='possibly_harmless_extra_action';break
                    idx=i;kind='wrong_requested_tool';break
                actual_args={k:v for k,v in action['args'].items() if k!='session_id'}
                exp_target=env.store.resolve_paper(env.session_id,args.get('ref')) if 'ref' in args else None
                actual_target=env.store.resolve_paper(env.session_id,actual_args.get('ref')) if 'ref' in args else None
                if exp_target and (not actual_target or not _same_paper(actual_target.id,exp_target.id)):
                    if actual_args.get('ref') is not None and _same_paper(str(actual_args['ref']),exp_target.id):error='correct_alias_backend_resolution_failure';break
                    future=t['expected_paper_ids'][i+1:]
                    if actual_target and any(p and _same_paper(actual_target.id,p) for p in future):error='possible_order_alternative';break
                    idx=i;kind='wrong_paper_reference';audit.update(expected_paper_id=exp_target.id,actual_paper_id=actual_target.id if actual_target else None);break
                defaults={k:v.default for k,v in inspect.signature(getattr(env,name)).parameters.items() if v.default is not inspect.Parameter.empty}
                mismatch=[k for k,v in args.items() if k!='ref' and not _match_arg_value(actual_args.get(k,defaults.get(k)),v,k)]
                if mismatch:
                    # Section aliases may express the same intent; absent resources must not teach a false negative.
                    if name=='get_paper_content' and 'section' in mismatch:error='section_alias_or_missing_snapshot';break
                    idx=i;kind='wrong_requested_parameter';audit['mismatched_parameters']=mismatch;break
                try:obs=_dispatch_environment_tool(env,name,actual_args)
                except Exception as exc:error='prefix_execution_failed_or_resource_gap';audit['prefix_error']=str(exc);break
                if str(obs)!=step['observation']:error='prefix_observation_replay_mismatch';break
                if step.get('model_observation')!=step['observation'][:1000]:error='bounded_observation_mismatch';break
                prefix.append(step)
            if error:omit(error);continue
            if idx is None or kind is None:omit('no_proven_wrong_decision');continue
            # Rejected replay can mutate state. Restore the certified prefix before executing chosen.
            _prepare_rollout_environment(env,t['id'],tasks)
            for h in prefix:
                a=json.loads(h['action']);assert str(_dispatch_environment_tool(env,a['name'],a['args']))==h['observation']
            if t.get('expected_terminal_mode')=='blocked' or kind=='malformed_action' and not expected:
                chosen_action='FINISH';chosen_thought=reference_terminal_thought(t)
            else:
                assert idx<len(expected)
                chosen_action={'name':expected[idx][0],'args':expected[idx][1]};chosen_thought='结合任务要求和当前会话状态，下一步应调用 '+expected[idx][0]
            suffix=[]
            try:
                for name,args in expected[idx:]:
                    obs=_dispatch_environment_tool(env,name,args)
                    suffix.append({'thought':'执行请求中尚未完成的操作','action':json.dumps({'name':name,'args':args},ensure_ascii=False),'observation':str(obs)})
                suffix.append({'thought':reference_terminal_thought(t),'action':'FINISH','observation':'任务完成'})
                metrics=extract_metrics(t,{'history':prefix+suffix},'cpu_certified',record['row_index'])
                assert is_strict_success(metrics)
            except Exception as exc:omit('chosen_suffix_not_certified');audit['chosen_error']=str(exc);continue
            visible=build_visible_setup_context(t)
            if prefix:visible+='\n\n'+'\n\n'.join(f"Thought: {h['thought']}\nAction: {h['action']}\nObservation: {h['model_observation']}" for h in prefix)
            prompt=get_react_prompt(t['task'],tools_desc,visible)
            rejected=turns[idx];raw_rejected=rejected
            normalization='none'
            if not rejected.lstrip().startswith('Thought:'):
                rejected='Thought:'+rejected;normalization='prepend_Thought_marker_only'
            chosen='Thought: '+chosen_thought+'\nAction: '+(chosen_action if isinstance(chosen_action,str) else json.dumps(chosen_action,ensure_ascii=False))
            identity=digest([t['id'],[(h['action'],h['model_observation']) for h in prefix],chosen_action,parse_react_action(raw_rejected)[1] if kind not in ['premature_finish','blocked_false_completion','malformed_action'] else _thought_of(raw_rejected)])
            audit.update(trajectory_step=idx,negative_type=kind,raw_rejected=raw_rejected,rejected_normalization=normalization,certified_prefix_steps=len(prefix),chosen_suffix_strict_success=True,chosen_metrics=metrics.to_dict(),decision_sha256=identity)
            if identity in seen:omit('duplicate_decision');audit['duplicate_of_row']=seen[identity];continue
            messages=[{'role':'user','content':prompt}]
            pair={'prompt':messages,'chosen':[{'role':'assistant','content':chosen}],'rejected':[{'role':'assistant','content':rejected}],'source_task_id':t['id'],'parent_task_id':t['parent_task_id'],'trajectory_step':idx,'negative_type':kind,'source_row_index':record['row_index'],'decision_sha256':identity,'source_raw_rejected_sha256':digest(raw_rejected),'pair_type':'real_preference'}
            lengths=[]
            for field in ['chosen','rejected']:
                ids=tok.apply_chat_template(messages+pair[field],tokenize=True)
                if hasattr(ids,'keys') and 'input_ids' in ids:ids=ids['input_ids']
                lengths.append(len(ids))
            if max(lengths)>=4096:omit('would_truncate_sequence');continue
            pair['max_sequence_tokens']=max(lengths);pairs.append(pair);seen[identity]=record['row_index'];audit['admitted']=True
        # One full original nonlinguistic expert pass; not reweighted toward failures.
        expert=[]
        for line in Path('data/repro_20261002/sft_train.jsonl').read_text().splitlines():
            x=json.loads(line)
            if x['augmentation']['task_variant'] or x['augmentation']['thought_variant']:continue
            assert x['parent_task_id'] in train and x['parent_task_id'] not in held
            expert.append({'prompt':[x['messages'][0]],'chosen':[x['messages'][1]],'rejected':[x['messages'][1]],'source_task_id':x['source_task_id'],'parent_task_id':x['parent_task_id'],'trajectory_step':x['trajectory_step'],'pair_type':'expert_replay','decision_sha256':x['sample_sha256']})
        assert len(expert)==398 and len({x['source_task_id'] for x in expert})==137 and {x['parent_task_id'] for x in expert}==train
        support={'unique_real_error_decisions':len(pairs),'error_semantic_tasks':len({x['source_task_id'] for x in pairs}),'error_parent_tasks':len({x['parent_task_id'] for x in pairs})}
        g=gate['support_gate'];passed=support['unique_real_error_decisions']>=g['minimum_unique_real_error_decisions'] and support['error_semantic_tasks']>=g['minimum_error_semantic_tasks'] and support['error_parent_tasks']>=g['minimum_error_parent_tasks']
        DATA.mkdir(parents=True,exist_ok=True)
        for name,rows in [('real_preferences.jsonl',pairs),('expert_replay.jsonl',expert),('combined_training.jsonl',expert+pairs)]:
            (DATA/name).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
        combined=expert+pairs
        (DATA/'chosen_control.jsonl').write_text(''.join(json.dumps({'prompt':x['prompt'],'completion':x['chosen'],'decision_sha256':x['decision_sha256'],'pair_type':x['pair_type']},ensure_ascii=False)+'\n' for x in combined))
        write(R/'negative-admission-audit.json',{'source_strict_failures':102,'support':support,'support_gate_passed':passed,'exclusions':dict(exclude),'negative_type_counts':dict(collections.Counter(x['negative_type'] for x in pairs)),'heldout_parent_overlap':0,'all_admitted_chosen_suffixes_strict_valid':True,'raw_decision_audits':decisions})
        sequence_lengths=[]
        for x in combined:
            for k in ['chosen','rejected']:
                ids=tok.apply_chat_template(x['prompt']+x[k],tokenize=True)
                if hasattr(ids,'keys') and 'input_ids' in ids:ids=ids['input_ids']
                sequence_lengths.append(len(ids))
        assert max(sequence_lengths)<4096
        manifest={'real_preference_rows':len(pairs),'expert_replay_rows':398,'combined_rows':len(combined),**support,'support_gate_passed':passed,'replay_semantic_tasks':137,'replay_parent_tasks':51,'heldout_parent_overlap':0,'max_sequence_tokens':max(sequence_lengths),'truncated_sequences':0,'synthetic_rejected_negatives':False,'expert_replay_rejected_equals_chosen':True,'expert_replay_interpretation':'Equality makes sigmoid-DPO constant log2 and zero preference gradient; CE still trains expert chosen. These are replay rows, not negative pairs.','prompt_scope':'Benchmark-format re-prompt of exact visible semantic prefix; original GRPO token IDs/EOS boundaries were not logged. No exact native token-prefix claim.','control':'Same combined prompt/chosen rows and order; no preference term.','sha256':{str(p):sha(p) for p in sorted(DATA.glob('*.jsonl'))}}
        write(R/'training-data-manifest.json',manifest);print(json.dumps(manifest,indent=2),flush=True)
        assert passed,'Data support gate failed; no downstream training permitted'
if __name__=='__main__':main()
