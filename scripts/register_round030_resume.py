#!/usr/bin/env python3
from pathlib import Path
import os,sys,datetime,shutil,subprocess
os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,'scripts')
from round029_online_core import R,read,write,sha,verify
N=R.parent/'round-030-2026-10-09'
def main():
 assert not (N/'finalization-amendment.json').exists();p=read(R/'experiment-plan.json');verify(p);s=read(N/'pre-resume-complete-audit.json');assert s['full_evaluation_complete'] and s['independently_rescored']==1320 and s['inference_jobs_closed']==26 and s['actual_completed_learning_steps']==128 and read(R/'experiment-status.json')['phase']=='failed_or_capacity_timeout'
 (N/'failure-preserved').mkdir();files=[R/'experiment-status.json',R/'controller.log',Path('scripts/finalize_round029_online.py')]
 for f in files:shutil.copy2(f,N/'failure-preserved'/f.name)
 comparisons=list(R.glob('*-vs-*.json'));assert len(comparisons)==16;frozen={str(f):sha(f) for f in comparisons+[R/'experiment-plan.json',R/'all-finals-before-inference.json']}
 for f,h in s['evidence_sha256'].items():assert sha(f)==h;frozen[f]=h
 for v in s['freeze']['models'].values():
  for f,h in v['sha256'].items():assert sha(f)==h;frozen[f]=h
 replacement=Path('scripts/finalize_round030_online.py');write(N/'finalization-amendment.json',{'registered_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_only':True,'learning_or_inference_changes':False,'cause':'Comparison CLI acceptsdev/iid_test/ood_test; frozen task-membership split keyfreshA/freshB was incorrectly forwarded asreport label. Alltraining/inference and1320 independent scores already complete.','only_semantic_change':'Map freshA/freshB display enum to iid_test, retaining identical task/oracle/raw files and original5ALLsuccess rules; reuse16 verified comparisons and create8 missing. Display enum affects report labels only, nottask selection.','preserved_failure_files_sha256':{str(f):sha(f) for f in (N/'failure-preserved').iterdir()},'replacement_source':str(replacement),'replacement_source_sha256':sha(replacement),'original_finalizer_sha256':sha(Path('scripts/finalize_round029_online.py')),'original_plan_sha256':sha(R/'experiment-plan.json'),'original_completed_artifacts_sha256':frozen,'new_learning_steps':0,'new_model_predictions':0,'planned_new_comparisons':8,'reused_comparisons':16,'no_retraining':True,'no_eval_rerun':True})
 with (N/'finalization-resume.log').open('w') as log:subprocess.run(['.venv-repro/bin/python',str(replacement)],stdout=log,stderr=subprocess.STDOUT,timeout=1200,check=True)
 print(read(N/'finalization-resume-result.json'))
if __name__=='__main__':main()
