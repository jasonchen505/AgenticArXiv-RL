#!/usr/bin/env python3
import os,sys,copy
from pathlib import Path
from round029_online_core import R,read,verify,worker_admission

def main():
 arm=sys.argv[1];plan=read(R/'experiment-plan.json');verify(plan);assert plan['learning_permitted'] and arm in plan['arms'] and worker_admission(int(os.environ.get('CUDA_VISIBLE_DEVICES','-1')));out=Path(plan['model_paths'][arm]).parent;assert not out.exists();sys.path.insert(0,'AgenticArxiv');from rl import train_grpo as module
 tasks=read(plan['train_tasks'])
 def load(task_set,split):
  assert task_set=='expanded' and split==plan['train_tasks']+':train_only';return copy.deepcopy(tasks)
 module._load_tasks=load;kwargs=dict(plan['native_main_kwargs']);kwargs.update(model=plan['source_model'],output_dir=str(out),seed=plan['arms'][arm]['seed'],lr=plan['arms'][arm]['lr']);module.main(**kwargs)
if __name__=='__main__':main()
