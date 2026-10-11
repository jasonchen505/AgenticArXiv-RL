import os,sys,argparse
from pathlib import Path
a=argparse.ArgumentParser();a.add_argument('--label',required=True);label=a.parse_args().label;assert label in ['source-plan-TRAIN-s49','mixed49-plan-TRAIN-s49']
root=Path('data/repro_round034')/label
os.environ.update(CUDA_VISIBLE_DEVICES='',STORE_BACKEND='memory',FIGURE_ANALYSIS_BACKEND='extractive')
for k,n in [('PDF_RAW_PATH','raw'),('PDF_TRANSLATED_PATH','translated'),('PDF_FIGURES_PATH','figures')]:os.environ[k]=str((root/n).resolve())
sys.path[:0]=['AgenticArxiv','scripts']
from recover_round035_frozen_plans import replay,S
from round034_core import R,read,write,verify,sha
from tools.bootstrap import require_all_tools
from round035_literal_error_log import install
import agents.base_agent as base
verify(read(S/'CPU-startup-amendment.json'));verify(read(S/'operation-amendment.json'));p=read(R/'experiment-plan.json');verify(p)
for f,h in read(R/'all-finals-before-inference.json')['sha256'].items():assert sha(f)==h
require_all_tools('CPU only: original frozen predictions, original startup paths')
try:base.log.error('tool error key={"paper_id":"fixture"}',exc_info=True)
except KeyError:pass
else:raise AssertionError('Original logging failure not reproduced')
log=install();log.error('tool error key={"paper_id":"fixture"}',exc_info=True);log.records=[];rs=replay(label,p)
write(S/(label+'-CPU-replay.json'),{'trials':rs,'literal_error_messages':log.records,'CPU_startup_config_matches_original_GPU_env':True,'new_model_predictions':0,'new_learning_steps':0})
print({'label':label,'full_trials':len(rs),'tool_error_messages':len(log.records)})
