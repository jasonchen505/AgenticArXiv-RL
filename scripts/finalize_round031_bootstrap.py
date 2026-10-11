"""Operational bootstrap only; frozen scientific finalizer and metrics unchanged."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']='';sys.path[:0]=['AgenticArxiv','scripts']
from tools.bootstrap import require_all_tools
require_all_tools('Frozen R031 CPU scoring finalizer')
from finalize_round031_study import main
if __name__=='__main__':main()
