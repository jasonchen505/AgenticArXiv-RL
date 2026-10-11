import os,sys,json,tempfile,copy,datetime,difflib
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',STORE_BACKEND='memory',FIGURE_ANALYSIS_BACKEND='extractive');sys.path[:0]=['AgenticArxiv','scripts']
from tools.bootstrap import require_all_tools
from benchmark.runner import BenchmarkRunner
from benchmark.metrics import extract_metrics,is_strict_success
from agents.prompt_templates import build_visible_setup_context
from arm_blind_observations import ArmBlindAgent,ArtifactAddressMap
from round031_study_core import R,read,write,sha
N=R.parent/'round-032-2026-10-10';require_all_tools('Arm-blind observation CPU integration')
class Fake:
 def __init__(self,a):self.answers=iter(a);self.requests=[]
 def chat_completions(self,**kwargs):self.requests.append(copy.deepcopy(kwargs));return {'choices':[{'message':{'content':next(self.answers)}}],'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}}
def run(t,mode,root):
 for k,n in [('PDF_RAW_PATH','raw'),('PDF_TRANSLATED_PATH','translated'),('PDF_FIGURES_PATH','figures')]:os.environ[k]=str(root/n)
 runner=BenchmarkRunner(agent_types=['regex'],offline=True,snapshot='data/repro_round024/trainA-snapshot.json');runner._reset_offline_trial_state();answers=['Thought: 按请求执行\nAction: '+json.dumps({'name':n,'args':a},ensure_ascii=False) for n,a in zip(t['expected_tools'],t['expected_tool_args'])]+['Thought: 请求全部完成\nAction: FINISH'];f=Fake(answers);agent=ArmBlindAgent(f,mode=mode,env=runner._tool_env(),side_effect_mgr=runner._side_effects(),max_iterations=t['max_iterations']+(0 if mode=='native' else 4));z=agent.run(t['task'],session_id='CPU_TRAIN_'+t['id'],initial_history=build_visible_setup_context(t));return z,f.requests
assert not (N/'observation-CPU-audit.json').exists();tasks=read('artifacts/reproduction/round-024-2026-10-07/train-tasks.json');count=0;steps=0;hist=0
with tempfile.TemporaryDirectory(dir='data/repro_round031',prefix='address-') as tmp:
 tmp=str(Path(tmp).resolve())
 for t in tasks:
  a,u=run(t,'native',Path(tmp)/'host-4-native-s47');b,v=run(t,'budget',Path(tmp)/'host-7-budget-s47');assert u==v and a['history']==b['history'];assert is_strict_success(extract_metrics(t,a,'CPU',0)) and is_strict_success(extract_metrics(t,b,'CPU',0));count+=1;hist+=len(a['history'])
  for x in a['artifact_observations']:
   assert ArmBlindAgent.__mro__[0].__name__=='ArmBlindAgent';assert ArtifactAddressMap({'raw':str(Path(tmp)/'host-4-native-s47/raw'),'translated':str(Path(tmp)/'host-4-native-s47/translated'),'figures':str(Path(tmp)/'host-4-native-s47/figures')}).public(x['actual_result'])==x['public_result'];steps+=1
  assert str(Path(tmp)) not in json.dumps(u,ensure_ascii=False)
 mapper=ArtifactAddressMap({'raw':str(Path(tmp)/'raw'),'translated':str(Path(tmp)/'translated'),'figures':str(Path(tmp)/'figures')});x={'text':'普通内容不变，'+str(Path(tmp)/'raw/a.pdf'),'paper_id':'2601.12345','p':str(Path(tmp)/'raw/a.pdf'),'nested':[str(Path(tmp)/'figures/p/1.png')],'outside':'/other/a.pdf'};y=mapper.public(x);assert y['text']==x['text'] and y['outside']==x['outside'] and y['paper_id']==x['paper_id'] and x['p']!=y['p'] and y['p']=='artifact://raw/a.pdf' and y['nested']==['artifact://figures/p/1.png']
# Diagnostic only: compare prior native/budget prompts until first differing request or response.
prior={};examples=[]
for seed in [45,46]:
 for group in read(R/'experiment-plan.json')['groups']:
  a=[json.loads(x) for x in (R/(group+'-native-s'+str(seed))/'traces.jsonl').read_text().splitlines()];b=[json.loads(x) for x in (R/(group+'-budget-s'+str(seed))/'traces.jsonl').read_text().splitlines()];c={'runs':len(a),'input_first':0,'response_first':0,'identical_shared_prefix':0,'path_only_first_input_diff':0}
  for u,v in zip(a,b):
   kind='identical_shared_prefix'
   for i,(x,y) in enumerate(zip(u['raw_result']['proposal_trace'],v['raw_result']['proposal_trace'])):
    if x['request']!=y['request']:
     kind='input_first';left=json.dumps(x['request'],ensure_ascii=False,sort_keys=True);right=json.dumps(y['request'],ensure_ascii=False,sort_keys=True);roota=str((Path('data/repro_round031')/(group+'-native-s'+str(seed))).resolve());rootb=str((Path('data/repro_round031')/(group+'-budget-s'+str(seed))).resolve());only=left.replace(roota,'artifact://scratch')==right.replace(rootb,'artifact://scratch');c['path_only_first_input_diff']+=only
     if only and len(examples)<2:examples.append({'task_id':u['task_id'],'trial':u['trial'],'seed':seed,'group':group,'first_differing_call':i+1,'changed_field':'download local_path exposed in Observation','native_physical_root':roota,'budget_physical_root':rootb,'diagnostic_only_not_training':True})
     break
    if x['response']['choices']!=y['response']['choices']:kind='response_first';break
   c[kind]+=1
  prior[str(seed)+'/'+group]=c
write(N/'observation-CPU-audit.json',{'passed':True,'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'TRAIN_only_new_code_tests':True,'scripted_expert_model_predictions':0,'paired64_raw_model_requests_identical':count,'paired64_public_histories_identical':count,'real_tool_results_preserved_and_recomputed':steps,'same_requested_semantics_and_outside_paths_unchanged':True,'only_structured_path_strings_within_configured_roots_relocated':True,'raw_physical_results_retained':True,'diagnostic_prior_control_comparison':prior,'diagnostic_examples':examples,'sha256':{f:sha(f) for f in ['scripts/arm_blind_observations.py','scripts/audit_round032_observations.py']}});print({'CPU_pairs':count,'expert_steps':steps,'prior_input_first':sum(c['input_first'] for c in prior.values()),'prior_path_only':sum(c['path_only_first_input_diff'] for c in prior.values())})
