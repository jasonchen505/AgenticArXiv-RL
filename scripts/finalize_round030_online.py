#!/usr/bin/env python3
"""Operational CPU-only resumption: map fresh report labels to accepted iid_test enum."""
from pathlib import Path
import datetime,subprocess
from round029_online_core import R,PY,collect,read,write,success_checks,sha
NEW=R.parent/'round-030-2026-10-09'
def main():
 amendment=read(NEW/'finalization-amendment.json');assert sha(Path(__file__))==amendment['replacement_source_sha256']
 for p,h in amendment['original_completed_artifacts_sha256'].items():assert sha(p)==h,p
 assert not (R/'online-outcome.json').exists();s=collect(full=True);p=read(R/'experiment-plan.json');checks=success_checks(s['counts']);assert set(checks)==set(p['success_rule']);comparisons=[];reused=created=0
 for group,g in p['groups'].items():
  for seed in [43,44]:
   for base,target in [('sft','rl'+str(seed)),('lr0_'+str(seed),'rl'+str(seed))]:
    d0=p['reused_evals'].get(group,{}).get(base,str(R/(group+'-'+base)));d1=str(R/(group+'-'+target));out=R/(base+'-vs-'+target+'-'+group+'.json')
    if out.exists():reused+=1
    else:
     label='iid_test' if group in ['freshA','freshB'] else g['split'].rsplit(':',1)[1];assert label in ['dev','iid_test','ood_test'];subprocess.run([PY,'scripts/compare_repro_evals.py','--base',d0,'--sft',d1,'--oracles',g['oracles'],'--output',str(out),'--base_label',base,'--sft_label',target,'--training_description','OriginalSFT nativeprogramGRPO versus same64nominal48visibleTRAIN lr0 control; scheduled64steps256samples same, realizedtokens/onpolicycontexts differ; fixed1e-6 beta.04 legacy reward, noselector','--evaluation_split',label],check=True);created+=1
    v=read(out);assert set(v['models'])=={base,target}
    for name,x in v['models'].items():
     c=s['counts'][group][name];assert x['authoritative_rescore_drift']==0 and x['runs']==c['runs'] and x['strict_success_rate']==c['strict']/c['runs'] and x['false_finish_rate']==c['false_finish']/c['runs'] and x['reliability']['pass^3']==c['all3_success_tasks']/(c['runs']//3)
    before=s['strict_per_task'][group][base];after=s['strict_per_task'][group][target];assert set(before)==set(after);assert v['task_gains']==sum(before[k]<3 and after[k]==3 for k in before) and v['task_losses']==sum(before[k]==3 and after[k]<3 for k in before);comparisons.append(str(out))
 assert len(comparisons)==24 and reused==16 and created==8;passed=all(checks.values());write(R/'online-outcome.json',dict(s,completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),comparisons=comparisons,rule_checks=checks,registered_rule_met=passed,fixed_recipe='completed_unpromoted' if passed else 'closed',no_automatic_promotion=True,scope=p['scope'],operational_amendment=str(NEW/'finalization-amendment.json')));write(R/'experiment-status.json',{'phase':'completed','updated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed_learning_steps':128,'completed_native_optimizer_steps':256,'success_rule_met':passed});write(NEW/'finalization-resume-result.json',{'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_only':True,'new_learning_steps':0,'new_model_predictions':0,'reused_comparisons_unchanged':16,'new_comparisons':8,'all_comparisons_verified':24,'registered_rule_met':passed,'rule_checks':checks,'original_inputs_unchanged':True});print('Finalized',checks)
if __name__=='__main__':main()
