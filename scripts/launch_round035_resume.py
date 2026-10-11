import os,sys,datetime,subprocess
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,'scripts')
from round034_core import R,PY,read,sha,write,verify
S=R.parent/'round-035-2026-10-10'
def main():
 assert not (S/'operation-launch.json').exists();verify(read(S/'operation-amendment.json'));receipt=read(S/'CPU-recovery-receipt.json');verify(receipt);assert receipt['CPU_replayed_full_trials']==96 and receipt['source96_history_semantic_metrics_prediction_usage_identical'];original=read(R/'launch.json');now=datetime.datetime.now(datetime.timezone.utc);deadline=datetime.datetime.fromisoformat(original['launched_at_utc'])+datetime.timedelta(seconds=original['finite_timeout_seconds']);remaining=int((deadline-now).total_seconds());assert remaining>0
 log=(S/'operation-controller.log').open('a');child=subprocess.Popen(['nohup','timeout','--kill-after=30',str(remaining),PY,'-u','scripts/resume_round035_operations.py'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);write(S/'operation-launch.json',{'launched_at_utc':now.isoformat(),'pid':child.pid,'remaining_original_deadline_seconds':remaining,'original_deadline_utc':deadline.isoformat(),'CPU_recovery_receipt_SHA256':sha(S/'CPU-recovery-receipt.json'),'operation_amendment_SHA256':sha(S/'operation-amendment.json'),'unstarted_inference_jobs_only':23,'new_training_jobs':0,'new_periodic_automation':False,'allowed_physical_gpus':[4,7]});print(read(S/'operation-launch.json'))
if __name__=='__main__':main()
