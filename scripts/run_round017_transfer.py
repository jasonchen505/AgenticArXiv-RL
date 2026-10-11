#!/usr/bin/env python3
"""Finite diagnostic confirmation; wait for CPUgold, then frozen GPU4 inference."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,shutil,subprocess,time
R=Path('artifacts/reproduction/round-017-2026-10-06');PY='.venv-repro/bin/python';MODELS=['sft','replay43','replay44'];D=Path('data/repro_round017')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def state(phase,**kw):write(R/'transfer-status.json',{'phase':phase,'updated_at_utc':stamp(),**kw})
def run(cmd,env=None):subprocess.run(cmd,env=env,check=True)
def main():
    with (R/'transfer.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        gate_deadline=time.monotonic()+1800
        while True:
            current=read(R/'world-build-status.json')
            if current['phase']=='world_ready':break
            if current['phase']=='failed':raise RuntimeError('CPU world gate failed; no new inference allowed: '+current.get('error',''))
            state('waiting_for_CPU_gold',world_phase=current['phase'],new_training_steps=0,model_runs_completed=0)
            if time.monotonic()>gate_deadline:raise TimeoutError('CPUgold finite wait expired; no new inference.')
            time.sleep(15)
        next_plan=read(R/'next-stage-plan.json');data=read(R/'world-data-manifest.json');assert data['gold_passed']==22 and data['paper_world_disjoint_base_ids']
        frozen={**next_plan['frozen_input_sha256'],**data['sha256']}
        frozen.update({p:sha(p) for p in ['scripts/run_round017_transfer.py','scripts/run_round016_gpu4_job.py','scripts/run_frozen_task_file.py','scripts/compare_repro_evals.py',str(R/'resource-admission-amendment.json')]})
        plan_path=R/'cross-world-experiment-plan.json'
        if not plan_path.exists():
            for p,h in frozen.items():assert sha(p)==h,p
            write(plan_path,{'registered_at_utc':stamp(),'sha256':frozen,'models':next_plan['models'],'model_order':MODELS,'tasks':22,'new_model_runs':198,'new_training_steps':0,'physical_gpu':4,'dtype':'bfloat16','seed':45,'repeat':3,'prompt_policy':'baseline','session_prefix':'bench_r20261003_075529','checkpoint_selection':'Allthreefinalsexistandhashfrozenbeforefirstnewworldinference;noseedselection.','gate_passed':22,'scope':'Paperdisjointdiagnostictransfer; observedprograms/parameters,extractivecaptionworld,not authorVLM.','success_interpretation':next_plan['success_interpretation']})
        plan=read(plan_path)
        for p,h in plan['sha256'].items():assert sha(p)==h,p
        for split in ['iid_test','ood_test']:
            for model in MODELS:
                label='transfer-'+model+'-'+split;record=R/(label+'-job.json');output=R/label
                if record.exists():assert read(record).get('exit_code')==0;continue
                deadline=time.monotonic()+1800
                while int(subprocess.check_output(['nvidia-smi','--id=4','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())<8192:
                    state('waiting_for_gpu4_headroom',next_job=label)
                    if time.monotonic()>deadline:raise TimeoutError('GPU4 headroom wait expired; do not preempt other process.')
                    time.sleep(30)
                scratch=D/'benchmark_scratch'
                if scratch.exists():shutil.rmtree(scratch)
                env=dict(os.environ)
                for key,name in [('PDF_RAW_PATH','pdf_raw'),('PDF_TRANSLATED_PATH','pdf_translated'),('PDF_FIGURES_PATH','pdf_figures')]:env[key]=str((scratch/name).resolve())
                cmd=[PY,'-u','scripts/run_frozen_task_file.py','--tasks',str(R/'world-tasks.json'),'--tasks_sha256',plan['sha256'][str(R/'world-tasks.json')],'--split',str(R/'world-splits.json')+':'+split,'--model',plan['models'][model],'--snapshot',str(D/'snapshot.json'),'--output',str(output),'--prefix',plan['session_prefix'],'--seed','45','--repeat','3']
                for p,h in plan['sha256'].items():assert sha(p)==h,p
                state('running',current_job=label,physical_gpu=4,new_training_steps=0)
                run([PY,'scripts/run_round016_gpu4_job.py','--record',str(record),'--log',str(R/(label+'.log')),'--gpu','4','--']+cmd,env)
        state('CPU_finalizing')
        for split in ['iid_test','ood_test']:
            for candidate in ['replay43','replay44']:
                target=R/('sft-vs-'+candidate+'-'+split+'.json')
                if not target.exists():run([PY,'scripts/compare_repro_evals.py','--base',str(R/('transfer-sft-'+split)),'--sft',str(R/('transfer-'+candidate+'-'+split)),'--oracles',str(R/'world-oracles.json'),'--output',str(target),'--base_label','sft','--sft_label',candidate,'--training_description','FrozenCEcandidatetransfer;paper-disjointworld,observedprograms,no additionaltraining','--evaluation_split',split])
        metrics={a:{} for a in MODELS};total_runs=0
        for split in ['iid_test','ood_test']:
            for candidate in ['replay43','replay44']:
                for model,m in read(R/('sft-vs-'+candidate+'-'+split+'.json'))['models'].items():
                    assert m['authoritative_rescore_drift']==0
                    metrics[model][split]=m
            for a in MODELS:
                summary=read(R/('transfer-'+a+'-'+split)/'summary.json');assert summary['error_count']==0;total_runs+=summary['sample_count']
        assert total_runs==198
        totals={a:sum(round(m['runs']*m['strict_success_rate']) for m in groups.values()) for a,groups in metrics.items()}
        effects={a:totals[a]-totals['sft'] for a in ['replay43','replay44']}
        for p,h in plan['sha256'].items():assert sha(p)==h,p
        write(R/'transfer-outcome.json',{'new_runs':198,'strict_totals':totals,'supervised_minus_source':effects,'both_seeds_beat_source':all(x>0 for x in effects.values()),'metrics':metrics,'no_automatic_promotion':True,'oldDev_source_retention_not_met_by_seed44':True,'new_training_steps':0,'scope':plan['scope']})
        state('completed',new_model_runs=198,new_training_steps=0)
if __name__=='__main__':
    try:main()
    except BlockingIOError:raise SystemExit('Anothertransfercontrollerownsthelock; no duplicate.')
    except Exception as exc:state('failed_or_wait_timeout',error=str(exc));raise
