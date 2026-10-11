#!/usr/bin/env python3
"""Freeze train-only semantic support for model-generated negative collection."""
from pathlib import Path
from datetime import datetime,timezone
import collections,hashlib,json,os,sys,tempfile
R=Path('artifacts/reproduction/round-013-2026-10-05');S=Path('data/repro_20261002/snapshot_curated_v1.json');SOURCE=Path('outputs/repro_20261003/sft_e1/final')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
    R.mkdir(parents=True,exist_ok=True);assert not (R/'collection-plan.json').exists()
    sys.path[:0]=['scripts','AgenticArxiv']
    from benchmark.tasks_expanded import get_expanded_tasks
    from generate_parametric_sft_data import build_parametric_tasks
    split=read('data/splits/v3_81.json')['split'];train=set(split['train']);held=set(split['dev']+split['iid_test']+split['ood_test'])
    tasks=[{**t,'parent_task_id':t['id']} for t in get_expanded_tasks() if t['id'] in train]
    tasks.extend({**d.spec.to_task(),'parent_task_id':d.parent_task_id} for d in build_parametric_tasks(include_t5=True))
    assert len(tasks)==137 and len({t['id'] for t in tasks})==137
    assert {t['parent_task_id'] for t in tasks}==train and not {t['parent_task_id'] for t in tasks}&held
    sft_ids={json.loads(s)['source_task_id'] for s in Path('data/repro_20261002/sft_train.jsonl').read_text().splitlines()}
    assert sft_ids=={t['id'] for t in tasks},'Use the original training semantics, not heldout-derived repairs'
    os.environ['STORE_BACKEND']='memory';results=[];oracles=[]
    with tempfile.TemporaryDirectory(prefix='round013-gold-') as temp:
        for key in ['PDF_RAW_PATH','PDF_TRANSLATED_PATH','PDF_TRANSLATED_LOG_PATH','PDF_CACHE_PATH','TRANSLATE_CACHE_PATH','PDF_FIGURES_PATH']:os.environ[key]=str(Path(temp)/key.lower())
        from benchmark.semantic_oracle import attach_expected_paper_ids,_call
        from benchmark.task_spec import reference_terminal_thought
        from benchmark.metrics import is_strict_success
        from rl.multiturn_env import AgenticArxivMultiTurnEnv
        from rl.reward import RewardCalculator
        from tools.bootstrap import require_all_tools
        require_all_tools('Round013 train-only gold validation');reward=RewardCalculator(curriculum_steps=0)
        for raw in tasks:
            t=attach_expected_paper_ids([raw],S)[0];env=AgenticArxivMultiTurnEnv(S);env.reset(task_id=t['id']);history=[]
            for setup in t.get('setup',[]):_call(env,setup['name'],setup.get('args',{}))
            for name,args in zip(t['expected_tools'],t['expected_tool_args']):
                obs=_call(env,name,args)
                if isinstance(obs,list):assert not any(x.get('_mock_env',{}).get('offline_fallback') for x in obs if isinstance(x,dict)),t['id']
                history.append({'thought':'执行训练契约标准步骤','action':json.dumps({'name':name,'args':args},ensure_ascii=False),'observation':json.dumps(obs,ensure_ascii=False,default=str)})
            history.append({'thought':reference_terminal_thought(t),'action':'FINISH','observation':'任务完成'})
            b,m=reward.compute_reward_breakdown(t,{'history':history});ok=is_strict_success(m) and b.total>=.99
            results.append({'task_id':t['id'],'parent_task_id':t['parent_task_id'],'strict_success':is_strict_success(m),'reward':b.total,'gold_valid':ok});oracles.append(t)
    write(R/'collection-gold-validation.json',{'semantic_tasks':137,'passed':sum(x['gold_valid'] for x in results),'model_inference':False,'results':results})
    assert all(x['gold_valid'] for x in results)
    write(R/'collection-tasks.json',tasks);write(R/'collection-oracles.json',oracles)
    old=read('artifacts/reproduction/round-004-2026-10-03/train-failures.json')
    old_tasks=collections.Counter(x['task_id'] for x in old['failures'])
    write(R/'existing-failure-support.json',{'old_train_rollouts':408,'strict_failures':old['strict_failures'],'failure_semantic_tasks':len(old_tasks),'failure_counts_by_task':dict(old_tasks),'fault_counts_overlapping':old['fault_counts_overlapping'],'strict_failure_is_not_automatically_valid_negative':True,'source_sha256':sha('artifacts/reproduction/round-004-2026-10-03/train-failures.json'),'decision':'Collect from all original137trainingsemantics instead of train on narrow seven-task failures or create more canned refusal texts.'})
    files=[str(R/'collection-tasks.json'),str(R/'collection-oracles.json'),str(R/'collection-gold-validation.json'),'data/splits/v3_81.json','data/repro_20261002/sft_train.jsonl',str(S),str(SOURCE/'adapter_model.safetensors'),'scripts/prepare_round013_collection.py','scripts/run_round013_collection.py','scripts/finalize_round013_collection.py','scripts/run_repro_job.py','AgenticArxiv/rl/train_grpo.py','AgenticArxiv/rl/multiturn_env.py','AgenticArxiv/rl/rollout_audit.py','AgenticArxiv/rl/grpo_reward.py','AgenticArxiv/rl/reward.py','AgenticArxiv/agents/prompt_templates.py','AgenticArxiv/benchmark/metrics.py','AgenticArxiv/benchmark/semantic_oracle.py']
    plan={'registered_at_utc':datetime.now(timezone.utc).isoformat(),'purpose':'Data collection for future real model-error preference/recovery data; no new learning objective or test-case rules.','source_model':str(SOURCE),'train_parent_tasks':51,'train_semantic_instances':137,'heldout_parent_overlap':0,'support_equals_original_sft_training_semantics':True,'groups_per_semantic_task':2,'generations_per_group':4,'planned_model_rollouts':1096,'native_loop_steps':274,'learning_rate':0.,'temperature':1.,'seed':46,'max_turns':6,'completion_token_budget':2048,'gpu':4,'allow_zero_variance':True,'dynamic_sampling':False,'loss_type':'grpo','scale_rewards':'group','beta':.04,'importance_sampling_level':'token','mask_truncated_completions':True,'reward_objective':'legacy','sha256':{p:sha(p) for p in files},'negative_admission_policy':['Recorded policy-authored output only; no heldout traces or handwritten substitute negatives.','Reconstruct the model-visible state and prove a valid chosen action at that same state; first divergence with certified valid prefix, or separately certified recovery branch.','Correct resolved paper aliases or harmless extra arguments are not automatically negative just because literal strict scoring fails.','Exclude ambiguous/successful equivalent actions and unresolved/truncated states; retain exclusion counts. Repeated trajectories or wrappers do not imply independent negative support.','Keep broad expert replay and matched chosen-only control for later training; no training until negative validity and support have been audited.'],'stop_synthetic_coefficient_search':True,'weight_invariance_requirement':'After collection,392source adapter tensors and frozen reference must remain exactly equal to original SFT; lr0 loop steps are not learning updates.','limits':['Same known caption world; original SFT train tasks, not new task/world generalization.','Model errors collected from one source SFT/one new samplingseed; support and diversity still require audit.','Gold/backend oracles only certify records, never enter the model prompt.']}
    write(R/'collection-plan.json',plan);print('Frozen train-only support:137gold/137,51parents,heldout overlap0,1096planned rollouts',flush=True)
if __name__=='__main__':main()
