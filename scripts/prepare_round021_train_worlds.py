#!/usr/bin/env python3
"""Frozen candidate metadata admission for two independent training paper worlds."""
from pathlib import Path
import datetime,hashlib,json,re
R=Path('artifacts/reproduction/round-021-2026-10-06');W=R.parent/'round-017-2026-10-06'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
base=lambda p:re.sub(r'v\d+$','',p['id'])
def main():
    assert not (R/'train-world-metadata-plan.json').exists()
    old=read(W/'next-stage-plan.json');held=read(W/'paper-bank.json')
    excluded=set(old['exclude_prior_world_base_ids'])|{base(p) for pool in held['pools'] for p in pool['papers']}
    sources=[W/'next-stage-plan.json',W/'paper-bank.json',W/'paper-bank-admitted.json',*sorted((W/'metadata-candidates').glob('pool-*.json')),Path(__file__)]
    assert len(sources)==14
    write(R/'train-world-metadata-plan.json',{'registered_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source':'Previously frozen official arXiv API50-candidate responses,2025 window;no new live query or model-ranked selection.','selection_rule':'For each world and pool take first10distinct eligible baseIDs in original response order. WorldB excludes allworldAunion;both exclude original231 and complete priorraw70,including unadmitted404 version.','used_model_outputs':False,'target_split':'Only137trainingsemantics/51trainingparents;metadata not yet training data. Heldout task labels, traces and program parents excluded.','new_training_steps':0,'GPU_jobs':0,'required_worlds':2,'required_pools_per_world':10,'required_rows_per_pool':10,'sha256':{str(p):sha(p) for p in sources}})
    specs=old['metadata_queries'];banks=[];blocked=set(excluded)
    for world in ['trainA','trainB']:
        pools=[]
        for i,spec in enumerate(specs):
            raw=read(W/'metadata-candidates'/f'pool-{i:02d}.json');assert raw['query']==spec['query']
            selected=[];seen=set()
            for p in raw['results']:
                identifier=base(p)
                if identifier in blocked or identifier in seen:continue
                assert p['published'][:4]=='2025'
                selected.append(p);seen.add(identifier)
                if len(selected)==10:break
            assert len(selected)==10,(world,spec['label'],len(selected))
            pools.append({**spec,'papers':selected,'source_sha256':sha(W/'metadata-candidates'/f'pool-{i:02d}.json')})
        ids={base(p) for pool in pools for p in pool['papers']}
        assert not ids&blocked
        bank={'world':world,'pools':pools,'unique_base_ids':len(ids),'overlap_with_original_or_prior_diagnostic_world':0,'selected_by_model_scores':False,'PDF_resource_gate_passed':False,'gold_gate_passed':False}
        write(R/(world+'-paper-bank.json'),bank);banks.append(bank);blocked|=ids
    write(R/'train-world-metadata-status.json',{'phase':'metadata_admitted','training_worlds':2,'pools_each':10,'papers_per_pool':10,'unique_papers_each':[x['unique_base_ids'] for x in banks],'prior_world_overlap':0,'between_new_world_overlap':0,'GPU_jobs':0,'new_training_steps':0,'PDFs_acquired':False,'gold_passed':False,'next_gate':'RealPDF/captionassets,resource-onlyadmission,retargettrainentityslotsandCPUgold137each. No learning until both worlds gold plus1239-row/token/split/provenance gate pass.'})
    write(R/'next-mainline-plan.json',{'registered_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'question':'Does training paper-world diversity improve Agent task robustness beyond simply repeating the same expert decisions for the same optimization budget?','closed_recipes':['real-error DPO+CE fixedbeta1:1','expert-observation-rendering fixed CE intervention'],'only_train_parents':51,'semantics_per_world':137,'source_model':'originalSFT','teacher_format':'OriginalCEexpert rendering,count-onlysearch/500-char generic;do not compound closed observation intervention.','arms':['repeat43','repeat44','world43','world44'],'rows_each':1239,'world_arm_composition':'398originalexperts+398trainAexperts+398trainBexperts+45unchangedoriginalmodel-errorchosenrows','repeat_control_composition':'398originalexperts repeated3times+45sameoriginalmodel-errorchosenrows','data_rows_and_chosen_CE_updates_match':True,'planned_steps_each':310,'training_dtype':'NF4/bf16','learning_rate':5e-6,'epochs':1,'microbatch':1,'gradient_accumulation':4,'seeds':[43,44],'dropout_disabled':True,'loss_coefficient_search':False,'current_stage':'CPUmetadata preparation only. Training not launched.','resource_gate':'Real PDFs pluscaption/T5complete;any deterministic resource substitutions preregistered beforemodeloutputs. No fabricated content.','training_gate':'All137gold eachnewworld;no heldoutparents;1239full orderedrows,completiontokens and no4096truncation audit;full scientificplan and blindconfirmationworld/taskprotocol freeze before training.','future_primary_confirmation':'At least two new paper-disjoint worlds excluded from original/prior diagnostic/newtrain banks,with predeclared heldoutparents/parameters. Frozen observedDev/worlds are diagnostic secondary;no case-by-case training patches.','success_direction':'Both training seeds beat same-seed repeat-budget controls on preregistered primary confirmation with retainedDev/falsefinish;fix numerical rule before training,otherwise close fixedworld-data recipe. Noautomaticpromotion.','allowed_physical_gpus':[4,7],'training_capacity_mib':23000,'training_requires_idle':True,'inference_capacity_mib':8192,'new_training_steps_this_interaction':0})
    print(json.dumps({'training_worlds':2,'unique_papers_each':[x['unique_base_ids'] for x in banks],'overlap':0,'training_launched':False}))
if __name__=='__main__':main()

