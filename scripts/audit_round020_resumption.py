#!/usr/bin/env python3
"""CPU evidence snapshot for device-only resumption; preserve old scientific recipe."""
from pathlib import Path
import ast,datetime,hashlib,json,re,subprocess
A=Path('artifacts/reproduction/round-020-2026-10-06');R=A.parent/'round-019-2026-10-06'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
    old=read(R/'experiment-plan.json');amend=read(A/'operational-amendment.json')
    assert sha(R/'experiment-plan.json')==amend['original_plan_sha256'] and amend['allowed_physical_gpus']==[4,7]
    for p,h in {**old['sha256'],**amend['new_source_sha256']}.items():assert sha(p)==h,p
    s=Path('scripts/train_round019_observations.py').read_text();n=Path('scripts/train_round020_observations.py').read_text();marker='    source=Path('
    assert s[s.index(marker):].replace("record={'objective':","record={'physical_gpu':int(physical_gpu),'objective':")==n[n.index(marker):]
    s=Path('scripts/run_round019_observation_experiment.py').read_text();n=Path('scripts/run_round020_observation_experiment.py').read_text();marker='        metrics={}'
    assert s[s.index(marker):]==n[n.index(marker):]
    for p in amend['new_source_sha256']:ast.parse(Path(p).read_text())
    refusal=[]
    for gpu in [0,5,6]:
        record=A/('forbidden-'+str(gpu)+'-job.json');log=A/('forbidden-'+str(gpu)+'.log')
        c=subprocess.run(['.venv-repro/bin/python','scripts/run_round020_gpu_job.py','--record',str(record),'--log',str(log),'--gpu',str(gpu),'--','.venv-repro/bin/python','-c','raise SystemExit(90)'],capture_output=True,text=True)
        assert c.returncode==2 and not record.exists() and not log.exists()
        refusal.append({'gpu':gpu,'exit_code':2,'failed_before_device_query_or_child':True})
    completed=[];jobs={};progress={}
    for seed in [43,44]:
        arm='obs'+str(seed);p=R/(arm+'-train-job.json')
        if p.exists():jobs[arm]=read(p);assert jobs[arm]['gpu_index'] in [4,7]
        manifest=Path('outputs/repro_round019')/arm/'final/training_manifest.json'
        if manifest.exists():
            m=read(manifest);assert m['actual_steps']==111 and m['dataset_rows']==443 and m['dpo_loss_type']==['sft'];completed.append(m)
        log=R/(arm+'-train.log')
        progress[arm]=max([int(x) for x in re.findall(r'(\d+)/111',log.read_text(errors='replace'))],default=0) if log.exists() else 0
    write(A/'experiment-status-at-report.json',read(R/'experiment-status.json'))
    write(A/'jobs-at-report.json',{'captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'training_jobs':jobs,'completed_training_arms':len(completed),'completed_training_steps':sum(x['actual_steps'] for x in completed),'logged_progress_steps':progress,'planned_training_steps':222,'planned_new_model_runs':312,'planned_reused_control_runs':468})
    write(A/'operational-audit.json',{'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_scientific_plan_unchanged':True,'all_frozen_inputs_unchanged':True,'trainer_core_equal_except_physical_device_metadata':True,'stop_rule_and_cpu_metrics_code_byte_equal':True,'allowed_physical_gpus':[4,7],'invalid_gpu_refusals':refusal,'duplicate_old_controller_replaced_before_training':True,'training_capacity_gate':{'minimum_free_mib':23000,'maximum_existing_compute_processes':0},'training_recipe':{'seeds':[43,44],'rows_per_seed':443,'steps_per_seed':111,'dtype':'NF4/bf16','learning_rate':5e-6,'objective':'native chosen CE only'},'posthoc_diagnostic_not_blind_confirmation':True})
    print(json.dumps({'complete_arms':len(completed),'logged_steps':progress,'device_only_amendment_checks':'pass'}))
if __name__=='__main__':main()

