#!/usr/bin/env python3
"""CPU-only closure after split-label CLI failure; frozen experiment unchanged."""
from pathlib import Path
import datetime,hashlib,json,os,shutil,subprocess
R=Path('artifacts/reproduction/round-008-2026-10-04');PY='.venv-repro/bin/python'
MODELS=['sft','hard43','gate43','hard44','gate44']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
    plan=read(R/'experiment-plan.json')
    for p,h in {**plan['sha256'],**plan['reused_evidence_sha256']}.items():assert sha(p)==h,p
    for a,h in plan['reference_model_hashes'].items():assert sha(Path(plan['evaluation_models'][a])/'adapter_model.safetensors')==h
    jobs=[read(p) for p in R.glob('*-job.json')]
    assert len(jobs)==18 and all(x['state']=='completed' and x['exit_code']==0 and x['gpu_index']==4 for x in jobs)
    assert not (R/'completion.json').exists()
    shutil.copy2(R/'controller-status.json',R/'controller-status-before-closure.json')
    pairs=[('sft',a) for a in ['hard43','gate43','hard44','gate44']]+[('hard43','gate43'),('hard44','gate44')]
    comparisons=[]
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
    for group in ['dev','previous-iid_test','previous-ood_test','fresh-iid_test','fresh-ood_test']:
        split=group if group=='dev' else group.split('-',1)[1]
        oracle='artifacts/reproduction/round-002-2026-10-02/task-oracles.json' if group=='dev' else str((Path('artifacts/reproduction/round-007-2026-10-04') if group.startswith('previous') else R)/'fresh-oracles.json')
        for base,candidate in pairs:
            leaf=lambda a:a+'-dev' if group=='dev' else group.split('-')[0]+'-'+a+'-'+split
            out=R/(base+'-vs-'+candidate+'-'+group+'.json')
            if not out.exists():
                cmd=[PY,'scripts/compare_repro_evals.py','--base',str(R/leaf(base)),'--sft',str(R/leaf(candidate)),'--oracles',oracle,'--output',str(out),'--base_label',base,'--sft_label',candidate,'--training_description','Fixed hard7,51steps,seed43/44; training-only blocked-reason cap; '+('EXPOSED development requests reused from Round007' if group.startswith('previous') or group=='dev' else 'Round008 new parameter requests, known paper world, observed heldout parents'),'--evaluation_split',split]
                subprocess.run(cmd,env=env,check=True,stdout=subprocess.DEVNULL)
            comparisons.append(str(out))
    write(R/'comparison-closure.json',{'state':'completed','scope':'CPU-only postprocessing, zero training/inference reruns','failure':'Original controller passed previous-iid_test to choices dev/iid_test/ood_test; parser rejected before comparison executed. All model jobs had already completed exit0.','correction':'Pass underlying iid_test/ood_test; keep previous/fresh scope explicit in filenames and training description. Frozen controller, plan, model weights, reward, scoring, tasks and raw outputs unchanged.','closed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'comparison_files':comparisons,'original_controller_sha256':sha('scripts/run_round008_experiment.py'),'closure_script_sha256':sha('scripts/close_round008_comparisons.py')})
    write(R/'completion.json',{'state':'completed','planned_new_benchmark_runs':510,'planned_reused_development_runs':270,'actual_steps_per_arm':51,'no_within_round_test_tuning':True,'cpu_only_comparison_closure':True})
    write(R/'controller-status.json',{'phase':'completed','updated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completion_method':'CPU-only comparison closure; preserved original failure status and controller log','all_model_jobs_exit0':True})
    print('Completed all30 comparisons; no GPU jobs rerun')
if __name__=='__main__':main()
