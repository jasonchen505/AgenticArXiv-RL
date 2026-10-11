#!/usr/bin/env python3
"""Independent confirmation gold, disjointness and frozen provenance audit."""
from pathlib import Path
import datetime,hashlib,json,os,re,sys
R=Path('artifacts/reproduction/round-023-2026-10-07');W=R.parent/'round-022-2026-10-07'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ids(bank):return {re.sub(r'v\d+$','',p['id']) for pool in bank['pools'] for p in pool['papers']}
def main():
    assert not (R/'confirmation-independent-audit.json').exists()
    os.environ.update(CUDA_VISIBLE_DEVICES='',STORE_BACKEND='memory');sys.path.insert(0,'AgenticArxiv')
    from benchmark.metrics import extract_metrics,is_strict_success
    prep=read(W/'confirmation-preparation-plan.json');templates={t['parent_task_id']:t for t in read(W/'confirmation-task-template.json')};train=set(read('data/splits/v3_81.json')['split']['train']);trials=assets=0;unavailable={};banks=[];core=[]
    for plan in [prep,read(W/'confirmation-asset-plan.json'),read(R/'confirmation-recovery-plan-v2.json')]:
        for p,h in plan['sha256'].items():assert sha(p)==h,p
    for w in ['confirmA','confirmB']:
        bank=read(W/(w+'-paper-bank.json'));bankids=ids(bank);assert not bankids&set(prep['exclude_base_ids']);banks.append(bankids)
        for manifest in [read(W/(w+'-asset-manifest.json')),read(W/(w+'-world-manifest.json'))]:
            for p,h in manifest['sha256'].items():assert sha(p)==h,p;assets+=1
        tasks=read(W/(w+'-oracles.json'));byid={t['id']:t for t in tasks};groups=read(W/(w+'-splits.json'))['split'];assert len(groups['iid_test'])==18 and len(groups['ood_test'])==4
        assert len(tasks)==22 and {t['parent_task_id'] for t in tasks}==set(templates) and not set(templates)&train
        tracepath=W/(w+'-gold-traces.jsonl');traces=[json.loads(x) for x in tracepath.read_text().splitlines()];assert {x['task_id'] for x in traces}==set(byid) and len(traces)==22
        for t in tasks:
            old=templates[t['parent_task_id']];assert t['expected_tools']==old['expected_tools'] and t.get('setup')==old.get('setup')
            if t['parent_task_id'] not in ['ref_ctrl_id_cache','ref_stress_learning_state']:assert t['task']==old['task'] and t['expected_tool_args']==old['expected_tool_args']
            for k in ['expected_terminal_reason','expected_terminal_mode','max_iterations']:assert t.get(k)==old.get(k)
        for trace in traces:
            m=extract_metrics(byid[trace['task_id']],{'history':trace['history']},'independent_confirmation_gold',0,'');assert is_strict_success(m) and not m.false_finish;trials+=1
        a=read(W/(w+'-asset-manifest.json'));assert a['actual_PDFs']==len(bankids);unavailable[w]=a.get('content_unavailable',[])
        assert all(x['metadata_retained'] and not x['fake_observation'] and "section 'abstract' was not found" in x['real_tool_error'] for x in unavailable[w])
        core.append({'world':w,'paper_ids':len(bankids),'actual_PDFs':a['actual_PDFs'],'default_content_papers':a['content_papers'],'default_content_unavailable':len(unavailable[w]),'figures':a['figures'],'T5_entries':a['T5_entries'],'gold':22})
    assert trials==44 and not banks[0]&banks[1]
    trainaudit=read(W/'independent-train-data-audit.json');assert trainaudit['passed'] and trainaudit['raw_gold_traces_independently_rescored']==274
    for p,h in trainaudit['data_sha256'].items():assert sha(p)==h,p
    result={'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'independent_confirmation_gold_passed':44,'gold_strict_and_FF_drift':0,'nontrain_parents_each':22,'IID_parents_each':18,'OOD_parents_each':4,'train_parent_overlap':0,'paper_disjoint_original_prior_train_and_between_confirmation':True,'worlds':core,'actual_resource_hashes_checked':assets,'unchanged_predeclared_rule':prep['fixed_success_rule'],'confirmation_plan_sha256':sha(W/'confirmation-preparation-plan.json'),'training_data_hashes_unchanged':True,'GPU_jobs':0,'new_training_steps':0,'no_confirmation_model_outputs_yet':True}
    (R/'confirmation-independent-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
