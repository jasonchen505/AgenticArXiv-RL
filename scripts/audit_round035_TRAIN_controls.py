import os,sys,json,datetime,subprocess,shutil,tarfile
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',STORE_BACKEND='memory',FIGURE_ANALYSIS_BACKEND='extractive');sys.path[:0]=['AgenticArxiv','scripts']
from round034_core import R,read,rows,write,sha
from finalize_round035_operational import collect
S=R.parent/'round-035-2026-10-10'
def main():
 assert not (S/'TRAIN-control-closure.json').exists();audit=collect(False);labels=['source-plan-TRAIN-s49','mixed49-plan-TRAIN-s49','lr0_49-plan-TRAIN-s49'];assert all(l in audit['counts'] for l in labels)
 left=rows(R/labels[0]/'predictions.jsonl');right=rows(R/labels[2]/'predictions.jsonl');assert len(left)==len(right)==48
 no_model=lambda x:{k:v for k,v in x.items() if k!='model'}
 prediction_parity={'pairs':48,'request_except_model_path_identical':sum(no_model(x['prediction']['request'])==no_model(y['prediction']['request']) for x,y in zip(left,right)),'choices_identical':sum(x['prediction']['response']['choices']==y['prediction']['response']['choices'] for x,y in zip(left,right)),'usage_identical':sum(x['prediction']['response']['usage']==y['prediction']['response']['usage'] for x,y in zip(left,right))}
 xs=rows(R/labels[0]/'traces.jsonl');ys=rows(R/labels[2]/'traces.jsonl');assert len(xs)==len(ys)==96
 parity={'pairs':96,'public_history_identical':sum(x['raw_result']['history']==y['raw_result']['history'] for x,y in zip(xs,ys)),'strict_and_FF_identical':sum(x['strict']==y['strict'] and x['metrics']['false_finish']==y['metrics']['false_finish'] for x,y in zip(xs,ys)),'token_usage_identical':sum(x['raw_result']['token_usage']==y['raw_result']['token_usage'] for x,y in zip(xs,ys))}
 mappings=sum(len(x['raw_result']['artifact_observations']) for label in audit['counts'] for x in rows(R/label/'traces.jsonl'));state=read(S/'operation-status.json');job_label=state.get('current_job');target=S/'TRAIN-control-snapshot';target.mkdir()
 if job_label:
  for suffix in ['-job.json','.log']:
   f=R/(job_label+suffix)
   if f.exists():shutil.copyfile(f,target/f.name)
 launch=read(S/'operation-launch.json');alive=subprocess.run(['kill','-0',str(launch['pid'])],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
 write(S/'TRAIN-control-closure.json',dict(audit,checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed_evidence_audit=True,source_lr0_prediction_parity=prediction_parity,source_lr0_full_trial_parity=parity,only_TRAIN_no_generalization_or_adoption=True,native_tool_result_mappings_recomputed=mappings,controller_status=state,controller_alive=alive,registered_six_ALL_can_no_longer_pass=True,source_SFT_remains_default=True))
 closed=S/'closed-R034'
 for label in audit['counts']:
  if not (closed/label).exists():
   shutil.copyfile(R/(label+'-job.json'),closed/(label+'-job.json'));shutil.copyfile(R/(label+'.log'),closed/(label+'.log'));shutil.copytree(R/label,closed/label)
 shutil.copyfile('scripts/audit_round035_TRAIN_controls.py',S/'source-copies/audit_round035_TRAIN_controls.py');files=[S/'TRAIN-control-closure.json',S/'source-copies/audit_round035_TRAIN_controls.py']+[f for f in target.rglob('*') if f.is_file()]+[f for label in audit['counts'] if label not in ['source-plan-TRAIN-s49','mixed49-plan-TRAIN-s49'] for f in (closed/label).rglob('*') if f.is_file()]+[closed/(label+suffix) for label in audit['counts'] if label not in ['source-plan-TRAIN-s49','mixed49-plan-TRAIN-s49'] for suffix in ['-job.json','.log']]
 archive=S.parent/'round035-TRAIN-control-closed.tar.gz';assert not archive.exists()
 with tarfile.open(archive,'w:gz') as tar:
  for f in files:tar.add(f,arcname=str(f.relative_to(S.parent)),recursive=False)
 print({'archive_SHA256':sha(archive),'full_trials':audit['full_trials_rescored'],'complete_units':audit['closed_inference_jobs'],'prediction_parity':prediction_parity,'trial_parity':parity,'controller':state})
if __name__=='__main__':main()
