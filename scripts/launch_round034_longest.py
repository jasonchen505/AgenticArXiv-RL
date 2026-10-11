import os,sys
os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,'scripts')
from round034_core import R,D,PY,read,write,sha
import run_round027_retention as h
def main():
 assert read(R/'preflight-result.json')['passed'];assert not (R/'checkpointed-preflight-job.json').exists();write(R/'checkpointed-preflight-plan.json',{'purpose':'Longest row with actual training gradient checkpointing; original first-row zero-step preflight preserved','optimizer_steps':0,'real_forward_backward_calls':1,'sha256':{f:sha(f) for f in ['scripts/check_round034_longest_batch.py','scripts/train_round034_plan.py','scripts/launch_round034_longest.py']}});h.state=lambda phase,**kw:write(R/'checkpointed-preflight-status.json',dict(phase=phase,**kw));gpu=h.wait_gpu(True,'checkpointed longest row');h.run_job([PY,'scripts/run_round020_gpu_job.py','--record',str(R/'checkpointed-preflight-job.json'),'--log',str(R/'checkpointed-preflight.log'),'--gpu',str(gpu),'--',PY,'-u','scripts/check_round034_longest_batch.py'],900,dict(os.environ,TMPDIR=str((D/'tmp').resolve())))
if __name__=='__main__':main()
