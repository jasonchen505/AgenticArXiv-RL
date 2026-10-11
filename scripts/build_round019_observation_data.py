#!/usr/bin/env python3
"""Train-only observation alignment through the unchanged benchmark execution path."""
from pathlib import Path
import collections,copy,datetime,hashlib,json,os,sys,tempfile
R=Path('artifacts/reproduction/round-019-2026-10-06');D=Path('data/repro_round019')
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
    R.mkdir(parents=True,exist_ok=True);D.mkdir(parents=True,exist_ok=True)
    assert not (R/'observation-alignment-audit.json').exists()
    oldpath=Path('data/repro_round014/certified_combined.jsonl');rows=[json.loads(x) for x in oldpath.read_text().splitlines()]
    tasks={t['id']:t for t in read('artifacts/reproduction/round-013-2026-10-05/collection-oracles.json')}
    split=read('data/splits/v3_81.json')['split'];train=set(split['train']);held=set(split['dev']+split['iid_test']+split['ood_test'])
    sourcepaths=[oldpath,Path('artifacts/reproduction/round-013-2026-10-05/collection-oracles.json'),Path('data/splits/v3_81.json'),Path('data/repro_20261002/snapshot_curated_v1.json'),Path('AgenticArxiv/agents/base_agent.py'),Path('AgenticArxiv/agents/prompt_templates.py'),Path(__file__)]
    write(R/'observation-alignment-plan.json',{'registered_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hypothesis':'Expert observations use count-only searches and 500-character generic results; current benchmark execution shows search titles and up to1000 chars. Audit and rebuild ALL137train semantic trajectories, preserve443ordered rows/chosen/rejected and45real-error prompts. No heldout cases or new-world papers in data builder.','CPU_only':True,'GPU_jobs':0,'new_training_steps':0,'scope':'Observation rendering intervention, not new task programs/state coverage or loss coefficients. No effect claim before matched two-seed evaluation.','sha256':{str(p):sha(p) for p in sourcepaths}})
    assert len(rows)==443
    experts=[x for x in rows if x['pair_type']=='expert_replay'];assert len(experts)==398
    assert {x['source_task_id'] for x in experts}==set(tasks) and {x['parent_task_id'] for x in experts}==train and not train&held
    os.environ.update(STORE_BACKEND='memory',CUDA_VISIBLE_DEVICES='',FIGURE_ANALYSIS_BACKEND='extractive')
    with tempfile.TemporaryDirectory(prefix='round019-observation-') as tmp:
        for key,name in [('PDF_RAW_PATH','raw'),('PDF_TRANSLATED_PATH','translated'),('PDF_FIGURES_PATH','figures')]:os.environ[key]=str(Path(tmp)/name)
        sys.path[:0]=['AgenticArxiv','scripts']
        from tools.bootstrap import require_all_tools
        require_all_tools('Round019 train-only observation audit')
        from models.store import use_memory_store
        from agents.agent_engine import ReActAgent
        from agents.side_effects import LocalSideEffectManager
        from agents.prompt_templates import get_react_prompt,build_visible_setup_context,format_tool_description
        from tools.tool_registry import registry
        from rl.env import MockArxivEnv
        from generate_sft_data import execute_expert_tool
        from benchmark.metrics import extract_metrics,is_strict_success
        from transformers import AutoTokenizer
        desc=format_tool_description(registry.list_tools());newby={};gold=[];observations=[];categories=collections.Counter()
        grouped=collections.defaultdict(list)
        for x in experts:grouped[x['source_task_id']].append(x)
        for task_id in sorted(grouped):
            sequence=sorted(grouped[task_id],key=lambda x:x['trajectory_step']);assert [x['trajectory_step'] for x in sequence]==list(range(len(sequence)))
            t=tasks[task_id];assert t['parent_task_id'] in train and t['parent_task_id'] not in held
            use_memory_store(reset=True)
            side=LocalSideEffectManager();env=MockArxivEnv('data/repro_20261002/snapshot_curated_v1.json',mode='replay',offline_download=True)
            agent=ReActAgent(None,side_effect_mgr=side,env=env,prompt_policy='baseline');agent.session_id='sft_spec_'+task_id
            for action in t.get('setup',[]):execute_expert_tool(env,side,action['name'],action.get('args',{}),agent.session_id)
            initial=build_visible_setup_context(t);history=[];oldhistory=[]
            for i,x in enumerate(sequence):
                oldprompt=x['prompt'][0]['content']
                oldrender='\n\n'.join([initial.strip()]+([agent.format_history(oldhistory)] if oldhistory else []))
                assert get_react_prompt(t['task'],desc,oldrender)==oldprompt,('Old expert template reconstruction',task_id,i)
                render='\n\n'.join([initial.strip()]+([agent.format_history(history)] if history else []))
                newprompt=get_react_prompt(t['task'],desc,render)
                nx=copy.deepcopy(x);nx['prompt'][0]['content']=newprompt
                nx['prompt_intervention']='benchmark_observation_rendering_only'
                newby[(task_id,i)]=nx
                chosen=x['chosen'][0]['content'];thought,actiontext=chosen.split('\nAction: ',1);thought=thought.removeprefix('Thought: ')
                if actiontext=='FINISH':obs='任务完成';name='FINISH'
                else:
                    action=json.loads(actiontext);name=action['name'];obs=agent._execute_with_side_effects(action)
                    assert not obs.startswith(('工具执行失败','错误:')),(task_id,i,obs)
                    categories[name]+=1
                history.append({'thought':thought,'action':actiontext,'observation':obs})
                if i+1<len(sequence):
                    oldobs=sequence[i+1]['prompt'][0]['content'].rsplit('\nObservation: ',1)[1].rstrip('\n')
                    oldhistory.append({'thought':thought,'action':actiontext,'observation':oldobs})
                    observations.append({'task_id':task_id,'parent_task_id':t['parent_task_id'],'step':i,'tool':name,'old_observation':oldobs,'benchmark_observation':obs,'changed':oldobs!=obs})
            metrics=extract_metrics(t,{'history':history},'cpu_observation_alignment',0,agent.session_id)
            assert is_strict_success(metrics),(task_id,metrics.to_dict())
            gold.append({'task_id':task_id,'strict_success':True,'expected_decisions':len(sequence)})
        output=[newby[(x['source_task_id'],x['trajectory_step'])] if x['pair_type']=='expert_replay' else x for x in rows]
        assert len(output)==443 and all(x['chosen']==y['chosen'] and x['rejected']==y['rejected'] and x['source_task_id']==y['source_task_id'] and x['parent_task_id']==y['parent_task_id'] and x['trajectory_step']==y['trajectory_step'] for x,y in zip(rows,output))
        assert all(x==y for x,y in zip(rows,output) if x['pair_type']!='expert_replay')
        tok=AutoTokenizer.from_pretrained('outputs/repro_20261003/sft_e1/final',local_files_only=True);lengths=[]
        for x in output:
            for k in ['chosen','rejected']:
                ids=tok.apply_chat_template(x['prompt']+x[k],tokenize=True)
                if hasattr(ids,'keys') and 'input_ids' in ids:ids=ids['input_ids']
                lengths.append(len(ids))
        assert max(lengths)<4096,max(lengths)
        target=D/'aligned_combined.jsonl';target.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in output))
        changes=sum(x['prompt']!=y['prompt'] for x,y in zip(rows,output))
        changedparents={x['parent_task_id'] for x,y in zip(rows,output) if x['prompt']!=y['prompt']}
        result={'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gold_passed':len(gold),'train_semantics':137,'train_parents':51,'heldout_parent_overlap':0,'ordered_rows':443,'expert_rows':398,'real_error_rows_unchanged':45,'changed_expert_prompt_rows':changes,'changed_expert_parent_tasks':len(changedparents),'chosen_rejected_and_order_unchanged':True,'non_observation_prompt_parts_reconstructed_and_equal':True,'new_task_programs':0,'new_world_data_used':False,'heldout_metrics_or_traces_used_to_select_examples':False,'max_sequence_tokens':max(lengths),'truncated_sequences':0,'observation_transition_count':len(observations),'observation_tool_counts':dict(categories),'renderer':'Unchanged ReActAgent._execute_with_side_effects + format_history + get_react_prompt,baseline','new_training_steps':0,'sha256':{str(target):sha(target)},'gold_results':gold,'observations':observations,'caveat':'Session/path handles differ across CPU scratch runs, alongside intended richer observations; no byte-level match to future sampled inference states.'}
        write(R/'observation-alignment-audit.json',result);write(R/'aligned-data-manifest.json',{k:result[k] for k in ['ordered_rows','expert_rows','real_error_rows_unchanged','gold_passed','max_sequence_tokens','truncated_sequences','sha256']})
        print(json.dumps({k:result[k] for k in ['gold_passed','changed_expert_prompt_rows','changed_expert_parent_tasks','max_sequence_tokens','new_training_steps']}))
if __name__=='__main__':main()

