import os,sys,json,datetime,collections
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';sys.path[:0]=['AgenticArxiv','scripts']
from tools.bootstrap import require_all_tools
from round031_study_core import read,write,sha
from run_round032_parity import N,verify
from arm_blind_observations import ArtifactAddressMap

def main():
 require_all_tools('R032 CPU infrastructure closure');assert not (N/'parity-outcome.json').exists();p=read(N/'parity-plan.json');verify(p);alltraces={};hashes={}
 for mode in p['modes']:
  job=read(N/('parity-'+mode+'-job.json'));assert job['exit_code']==0 and job['gpu_index'] in [4,7];out=N/('parity-'+mode);traces=[json.loads(x) for x in (out/'traces.jsonl').read_text().splitlines()];assert len(traces)==64;root=Path('data/repro_round032')/mode;mapper=ArtifactAddressMap({n:str(root/n) for n in ['raw','translated','figures']});alltraces[mode]={x['task_id']:x for x in traces};hashes[str(out/'traces.jsonl')]=sha(out/'traces.jsonl')
  for x in traces:
   for z in x['raw_result']['artifact_observations']:assert mapper.public(z['actual_result'])==z['public_result']
 a,b=[alltraces[m] for m in p['modes']];assert set(a)==set(b);c=collections.Counter();pairs=[]
 for tid,x in a.items():
  y=b[tid];u=x['raw_result']['proposal_trace'];v=y['raw_result']['proposal_trace'];kind='identical_shared_prefix';call=None
  for i,(q,r) in enumerate(zip(u,v)):
   if q['request']!=r['request']:kind='unexplained_input_first';call=i+1;break
   if q['response']['choices']!=r['response']['choices']:kind='model_response_first';call=i+1;break
  c[kind]+=1;pairs.append({'task_id':tid,'shared_prefix_class':kind,'first_difference_call':call,'native_calls':len(u),'budget_calls':len(v)})
 result={'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed_input_parity_gate':c['unexplained_input_first']==0,'classification':c,'pairs':pairs,'full_trials':128,'nominal_TRAIN_ids':64,'unique_visible_TRAIN_contracts':48,'learning_steps':0,'model_efficacy_claim':False,'raw_physical_results_preserved':True,'raw_sha256':hashes,'plan_sha256':sha(N/'parity-plan.json'),'no_automatic_heldout_or_learning_launch':True};write(N/'parity-outcome.json',result);print(result['classification'])
if __name__=='__main__':main()
