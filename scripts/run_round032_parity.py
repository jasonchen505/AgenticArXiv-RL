import os,sys,json,argparse
from pathlib import Path
sys.path[:0]=['AgenticArxiv','scripts']
from round031_study_core import PY,read,write,sha
from benchmark.runner import BenchmarkRunner
from arm_blind_observations import ArmBlindAgent
N=Path('artifacts/reproduction/round-032-2026-10-10')
def verify(p):
 for f,h in p['sha256'].items():assert sha(f)==h,f
class Runner(BenchmarkRunner):
 def __init__(self,*args,mode,**kwargs):self.mode=mode;super().__init__(*args,**kwargs)
 def _create_agent(self,agent_type,max_iterations=None):
  assert agent_type=='regex';return ArmBlindAgent(self.llm_client,mode=self.mode,side_effect_mgr=self._side_effects(),env=self._tool_env(),max_iterations=(max_iterations or 5)+(0 if self.mode=='native' else 4),llm_extra=self.llm_extra,prompt_policy='baseline')
def main():
 a=argparse.ArgumentParser();a.add_argument('--mode',choices=['native','budget'],required=True);args=a.parse_args();p=read(N/'parity-plan.json');verify(p);out=N/('parity-'+args.mode);assert not out.exists();out.mkdir();from tools.bootstrap import require_all_tools;require_all_tools('Arm-blind TRAIN input sanity');from benchmark.report import BenchmarkReport
 tasks=sorted(read(p['tasks']),key=lambda t:t['id']);runner=Runner(mode=args.mode,agent_types=['regex'],repeat=1,model=p['source_model'],session_prefix='bench_r20261003_075529',offline=True,snapshot=p['snapshot'],llm_backend='transformers',local_device='cuda',local_dtype='bfloat16',generation_seed=47,prompt_policy='baseline');results=runner.run_all(tasks);assert len(results)==64 and not any(x.error for x in results);traces=[]
 for x in results:
  raw=x.raw_result;assert len(raw['proposal_trace'])==raw['LLM_calls']<=raw['max_LLM_calls'];assert 'data/repro_round032' not in json.dumps([y['request'] for y in raw['proposal_trace']],ensure_ascii=False);traces.append({'task_id':x.task_id,'trial':x.trial,'session_id':x.session_id,'history':raw['history'],'raw_result':raw})
 BenchmarkReport([x.metrics for x in results],model=p['source_model'],errors=[]).save_all(str(out));(out/'traces.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False,default=str)+'\n' for x in traces));write(out/'execution_protocol.json',{'TRAIN_only':True,'infra_sanity_not_efficacy':True,'mode':args.mode,'seed':47,'repeat':1,'source_model':p['source_model'],'stable_public_artifact_addresses':True,'actual_physical_results_preserved':True,'no_oracle_in_model_prompt':True,'original_scorer_unchanged':True});print('Completed 64 TRAIN infrastructure sanity trials')
if __name__=='__main__':main()
