"""Finite offline Qwen3.5 replay; public inputs only, full raw costs preserved."""
import argparse,json,os,subprocess,hashlib
from pathlib import Path
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['bank','catalog','tasks','model-dir','out']:p.add_argument('--'+name,required=True)
 p.add_argument('--arm',choices=['on_compiled','on_two','off_compiled'],required=True)
 p.add_argument('--gpu',type=int,choices=[4,7],required=True)
 p.add_argument('--inject-failure',action='append',default=[],metavar='TASK_ID')
 args=p.parse_args();out=Path(args.out);assert not out.exists(),'Refuse existing output; no answer reruns'
 os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
 free=int(subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>=13000,'Insufficient admission memory; other jobs untouched'
 import torch
 from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
 from types import SimpleNamespace
 from .execution import run
 torch.cuda.set_per_process_memory_fraction(.5,0)
 modelpath=Path(args.model_dir);tok=AutoTokenizer.from_pretrained(modelpath,local_files_only=True);model=Qwen3_5ForConditionalGeneration.from_pretrained(modelpath,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval();client=SimpleNamespace(tokenizer=tok,model=model)
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 thinking=args.arm.startswith('on_');native={'arm':'qwen35_on' if thinking else 'qwen35_off','model':'Qwen/Qwen3.5-4B','thinking':thinking,'generation_cap':8192 if thinking else 256,'sampling':{'temperature':1.0 if thinking else .7,'top_p':.95 if thinking else .8,'top_k':20,'min_p':0.,'presence_penalty':1.5,'repetition_penalty':1.0}}
 arm={'name':'qwen35_'+args.arm,'native':native,'mode':'two_call' if args.arm=='on_two' else 'compiled','selector':'old'};read=lambda f:json.loads(Path(f).read_text());bank,cat,tasks=read(args.bank),read(args.catalog),read(args.tasks)
 out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('x') as f:
  for task in tasks:
   task={'split':'authored_offline_replay',**task};trace=run(client,arm,{'fault_ids':args.inject_failure},bank,cat,task);trace['GPU']=args.gpu
   f.write(json.dumps(trace,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno());print(task['id'],trace['LLM_calls'],trace['token_usage']['total_tokens'],trace['errors'],flush=True)
 out.with_suffix(out.suffix+'.receipt.json').write_text(json.dumps({'complete':True,'tasks':len(tasks),'arm':args.arm,'GPU':args.gpu,'traces_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'new_optimizer_steps':0,'no_private_rubric_or_answers_in_inputs':True,'input_sha256':{n:hashlib.sha256(Path(getattr(args,n)).read_bytes()).hexdigest() for n in ['bank','catalog','tasks']}},indent=2)+'\n')
if __name__=='__main__':main()
