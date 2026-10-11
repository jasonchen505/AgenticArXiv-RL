"""Match each original worker's startup config in the CPU-only replay fixture."""
import os,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';sys.path[:0]=['AgenticArxiv','scripts']
import recover_round035_frozen_plans as original
from config import settings
from round034_core import R,D,read,write,verify,sha
S=R.parent/'round-035-2026-10-10'
saved_replay=original.replay
def aligned_replay(label,p):
 for field,n in [('pdf_raw_path','raw'),('pdf_translated_path','translated'),('pdf_figures_path','figures')]:setattr(settings,field,str((D/label/n).resolve()))
 return saved_replay(label,p)
def main():
 amend=read(S/'CPU-context-amendment.json');verify(amend);original.replay=aligned_replay;original.main();receipt=read(S/'CPU-recovery-receipt.json');receipt['CPU_context_fixture_matches_original_GPU_startup_config']=True;receipt['CPU_context_amendment_SHA256']=sha(S/'CPU-context-amendment.json')
 for f in [S/'CPU-context-amendment.json',Path('scripts/recover_round035_aligned_context.py')]:receipt['sha256'][str(f)]=sha(f)
 write(S/'CPU-recovery-receipt.json',receipt)
if __name__=='__main__':main()
