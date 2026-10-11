#!/usr/bin/env python3
"""Pre-learning paper-blind confirmation metadata and fixed contract template."""
from pathlib import Path
import copy,datetime,hashlib,json,os,re,subprocess
R=Path('artifacts/reproduction/round-022-2026-10-07');OLD=R.parent/'round-021-2026-10-06';W=R.parent/'round-017-2026-10-06'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def base(p):return re.sub(r'v\d+$','',p['id'])
def status(phase,**kw):write(R/'confirmation-status.json',dict(phase=phase,updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),GPU_jobs=0,new_training_steps=0,**kw))
def main():
    assert not (R/'confirmation-preparation-plan.json').exists()
    oldplan=read(W/'next-stage-plan.json');banks=[read(W/'paper-bank.json')]+[read(OLD/(w+'-paper-bank.json')) for w in ['trainA','trainB']]
    blocked=set(oldplan['exclude_prior_world_base_ids'])|{base(p) for b in banks for pool in b['pools'] for p in pool['papers']}|{base(p) for w in ['trainA','trainB'] for p in read(R/(w+'-CL-reserve.json'))}
    taskpath=R.parent/'round-014-2026-10-05/fresh-tasks.json';splitpath=R.parent/'round-014-2026-10-05/fresh-splits.json';templates=read(taskpath);groups=read(splitpath)['split'];training=read('data/splits/v3_81.json')['split']['train']
    assert len(templates)==22 and not {t['parent_task_id'] for t in templates}&set(training)
    for t in templates:
        t['id']=t['id'].replace('fresh014_','confirm022_')
        for a,b in [(23,26),(24,27),(25,28)]:t['task']=t['task'].replace('最近'+str(a)+'天','最近'+str(b)+'天')
        for action in t.get('setup',[])+[{'name':n,'args':a} for n,a in zip(t['expected_tools'],t['expected_tool_args'])]:
            args=action['args']
            if args.get('days') in [23,24,25]:args['days']+=3
            if action['name']=='search_arxiv_papers':args['max_results']=8;t['task']=t['task'].replace('最多7篇','最多8篇')
            if action['name']=='summarize_paper':args['max_words']=180;t['task']=t['task'].replace('160词','180词')
        if t['parent_task_id']=='infeasible_index_out_of_range':t['task']=t['task'].replace('第8篇','第9篇')
    write(R/'confirmation-task-template.json',templates);write(R/'confirmation-template-splits.json',{'split':{k:[x.replace('fresh014_','confirm022_') for x in v] for k,v in groups.items()}})
    queries={}
    for world,year in [('confirmA','2023'),('confirmB','2022')]:
        queries[world]=[{**s,'query':s['query'].replace('202501010000 TO 202512312359',year+'01010000 TO '+year+'12312359')} for s in oldplan['metadata_queries']]
    sources=[Path(__file__),taskpath,splitpath,Path('data/splits/v3_81.json'),W/'next-stage-plan.json',W/'paper-bank.json',OLD/'next-mainline-plan.json',R/'CL12-resource-amendment.json',R/'confirmation-task-template.json',R/'confirmation-template-splits.json',Path('scripts/prepare_round017_paper_bank.py')]+[OLD/(w+'-paper-bank.json') for w in ['trainA','trainB']]+[R/(w+'-CL-reserve.json') for w in ['trainA','trainB']]
    write(R/'confirmation-preparation-plan.json',{'registered_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'queries':queries,'exclude_base_ids':sorted(blocked),'selection_rule':'Per pool first10distinct eligible IDs in frozen official response order. confirmB additionally excludes full confirmA union. Stop if insufficient or required resources fail; no model-based replacement.','primary_protocol':'Two paper-blind worlds,22heldout parents each,18IID+4OOD,repeat3,seed45,baseline,bf16;new26/27/28day parameters,keyword8,summary180,invalidordinal9with4papers. Same task programs,not unseen programs/not guaranteed pretraining unseen. Positive title bind AIordinal3 via unique full title;ID bind CVordinal1. Resource-only permutation for AIordinal5figure>=1 andCRordinal3nonemptyAbstract before oracle/model outputs.','models':['sourceSFT','repeat43','repeat44','world43','world44'],'new_evaluation_runs_planned':756,'frozen_Dev_source_reused':24,'rows_each_future':1239,'steps_each_future':310,'four_learning_arms':['repeat43','repeat44','world43','world44'],'fixed_success_rule':{'each_seed_each_world_strict_gain_vs_matched_repeat_at_least':2,'Dev_strict_world_each_seed_at_least_source':10,'Dev_strict_world_each_seed_at_least_matching_repeat':True,'IID_false_finish_each_seed_each_world_no_greater_than_matching_repeat':True,'Dev_false_finish_each_seed_no_greater_than_matching_repeat':True,'Dev_false_finish_each_seed_at_most_source':6,'all_conditions_required':True},'failure_action':'Close fixed world-data recipe. No coefficient/case/seed/checkpoint selection, no automatic promotion. Only final310step adapters; freeze all4before first candidate inference.','scientific_plan_state':'Confirmation preparation rule fixed before learning and model predictions. Full training protocol with final dataset/tool/world/model SHAs must be registered after resources+gold+independent data gates, before learning. This plan grants no learning yet.','GPU_jobs':0,'new_training_steps':0,'allowed_physical_gpus':[4,7],'sha256':{str(p):sha(p) for p in sources}})
    counts={}
    for w in ['confirmA','confirmB']:
        folder=R/(w+'-metadata-candidates');folder.mkdir(exist_ok=True);pools=[]
        for i,spec in enumerate(queries[w]):
            status('querying_confirmation_metadata',world=w,completed_pools=i,completed_worlds=len(counts))
            out=folder/f'pool-{i:02d}.json'
            subprocess.run(['.venv-repro/bin/python','scripts/prepare_round017_paper_bank.py','--query',spec['query'],'--output',str(out)],timeout=90,check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
            raw=read(out);assert raw['query']==spec['query'];selected=[];seen=set()
            for p in raw['results']:
                if base(p) in blocked or base(p) in seen:continue
                selected.append(p);seen.add(base(p))
                if len(selected)==10:break
            assert len(selected)==10,(w,spec['label'],len(selected))
            pools.append({**spec,'papers':selected,'source_sha256':sha(out)})
        ids={base(p) for pool in pools for p in pool['papers']};assert not ids&blocked
        write(R/(w+'-paper-bank.json'),{'world':w,'pools':pools,'unique_base_ids':len(ids),'excluded_overlap':0,'selected_by_model_scores':False,'resource_gate_passed':False,'gold_gate_passed':False});blocked|=ids;counts[w]=len(ids)
    status('confirmation_metadata_ready_not_gold',unique_papers=counts,gold_passed=False,model_outputs=0)
if __name__=='__main__':
    try:main()
    except Exception as e:status('failed',error=str(e));raise
