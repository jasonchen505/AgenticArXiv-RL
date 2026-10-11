#!/usr/bin/env python3
from pathlib import Path
import json,hashlib,datetime
R=Path('artifacts/reproduction/round-017-2026-10-06');A=Path('artifacts/reproduction/round-018-2026-10-06');D=Path('data/repro_round017')
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
write=lambda p,x:Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
old=read(A/'snapshot-before-T5-repair.json');new=read(D/'snapshot.json')
assert {k:v for k,v in old.items() if k!='search_arxiv_papers'}=={k:v for k,v in new.items() if k not in ['search_arxiv_papers','analyze_figure']}
assert sorted(json.dumps(v,sort_keys=True) for v in old['search_arxiv_papers'].values())==sorted(json.dumps(v,sort_keys=True) for v in new['search_arxiv_papers'].values())
assert len(new['analyze_figure'])==618
gold=read(R/'world-gold-validation.json');assert gold['passed']==gold['tasks']==22 and all(x['passed'] for x in gold['results'])
bank=read(R/'paper-bank-admitted.json');plan=read(R/'cross-world-experiment-plan.json')
assert list(plan['models'])==['sft','replay43','replay44'] and plan['physical_gpu']==4 and plan['new_model_runs']==198 and plan['new_training_steps']==0
assert sha('scripts/run_round017_transfer.py')==read(R/'transfer-launch-plan.json')['source_sha256']
for p,h in plan['sha256'].items():assert sha(p)==h,p
baseIDs={p['id'].split('v')[0] for pool in bank['pools'] for p in pool['papers']}
assert len(baseIDs)==69 and not baseIDs & set(read(R/'next-stage-plan.json')['exclude_prior_world_base_ids'])
assert len(list((D/'pdf_raw').glob('*.pdf')))==40
figures=[f for entry in new['extract_paper_figures'].values() for f in entry['result']['figures']]
assert len(figures)==206 and all(Path(f['path']).is_file() for f in figures)
write(A/'readiness-audit.json',{'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gold_passed':22,'metadata_unique_papers':69,'PDF_files':40,'figure_metadata':206,'T5_records':618,'prior_world_paper_overlap':0,'original_world_entries_unchanged_except_T5_addition_and_keyword_key_canonicalization':True,'all_frozen_inputs_unchanged':True,'original_transfer_controller_unchanged':True,'gold_assets_unchanged':True,'new_training_steps':0,'planned_model_runs':198,'physical_gpu':4,'scope':'paper-disjoint, observed task programs/parameters; caption backend, not original VLM','DPO_branch':'closed'})
records={p.name:read(p) for p in R.glob('transfer-*-job.json')}
assert all(x['gpu_index']==4 for x in records.values())
complete=[x for x in records.values() if x.get('exit_code')==0]
write(A/'transfer-status-at-report.json',read(R/'transfer-status.json'))
write(A/'world-build-status-at-report.json',read(R/'world-build-status.json'))
write(A/'jobs-at-report.json',{'captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'records':records,'completed_jobs':len(complete),'total_jobs':6,'completed_runs':sum(read(R/name.replace('-job.json','')/'summary.json')['sample_count'] for name,rec in records.items() if rec.get('exit_code')==0),'planned_runs':198})
if (R/'transfer-outcome.json').exists():write(A/'transfer-outcome-at-report.json',read(R/'transfer-outcome.json'))

