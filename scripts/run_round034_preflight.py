import os,sys,random,subprocess,datetime
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';sys.path[:0]=['AgenticArxiv','scripts']
from round034_core import R,D,PY,read,rows,sha,write
import run_round027_retention as h
def state(phase,**kw):write(R/'preflight-status.json',dict(phase=phase,updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw))
def main():
 assert read(R/'supervision-CPU-audit.json')['passed'];assert not (R/'preflight-plan.json').exists();old=read(R.parent/'round-033-2026-10-10/pilot-plan.json');files=dict(old['sha256'])
 for f in [D/'mixed-supervision.jsonl',R/'supervision-CPU-audit.json',Path('scripts/train_round034_plan.py'),Path('scripts/round034_core.py'),Path('scripts/run_round034_preflight.py')]:files[str(f)]=sha(f)
 order=list(range(827));random.Random(49).shuffle(order);write(R/'preflight-plan.json',{'registered_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_model':old['source_model'],'training_data':str(D/'mixed-supervision.jsonl'),'training_order':order,'learning_permitted':False,'optimizer_steps':0,'real_forward_backward_calls':1,'allowed_physical_gpus':[4,7],'sha256':files});h.state=state;gpu=h.wait_gpu(True,'new completion-only CE preflight');env=dict(os.environ,TMPDIR=str((D/'tmp').resolve()));state('running',physical_gpu=gpu);h.run_job([PY,'scripts/run_round020_gpu_job.py','--record',str(R/'preflight-job.json'),'--log',str(R/'preflight.log'),'--gpu',str(gpu),'--',PY,'-u','scripts/train_round034_plan.py','--arm','mixed49','--preflight'],900,env);assert read(R/'preflight-result.json')['passed'];state('completed')
if __name__=='__main__':main()
