#!/usr/bin/env python3
import datetime,subprocess
from round029_online_core import R,PY,collect,read,write,success_checks

def main():
 assert not (R/'online-outcome.json').exists();s=collect(full=True);p=read(R/'experiment-plan.json');checks=success_checks(s['counts']);assert set(checks)==set(p['success_rule']);comparisons=[]
 for group,g in p['groups'].items():
  for seed in [43,44]:
   for base,target in [('sft','rl'+str(seed)),('lr0_'+str(seed),'rl'+str(seed))]:
    d0=p['reused_evals'].get(group,{}).get(base,str(R/(group+'-'+base)));d1=str(R/(group+'-'+target));out=R/(base+'-vs-'+target+'-'+group+'.json');subprocess.run([PY,'scripts/compare_repro_evals.py','--base',d0,'--sft',d1,'--oracles',g['oracles'],'--output',str(out),'--base_label',base,'--sft_label',target,'--training_description','OriginalSFT nativeprogramGRPO versus same64nominal48visibleTRAIN lr0 control; scheduled64steps256samples same, realizedtokens/onpolicycontexts differ; fixed1e-6 beta.04 legacy reward, noselector','--evaluation_split',g['split'].rsplit(':',1)[1]],check=True);assert all(x['authoritative_rescore_drift']==0 for x in read(out)['models'].values());comparisons.append(str(out))
 assert len(comparisons)==24;passed=all(checks.values());write(R/'online-outcome.json',dict(s,completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),comparisons=comparisons,rule_checks=checks,registered_rule_met=passed,fixed_recipe='completed_unpromoted' if passed else 'closed',no_automatic_promotion=True,scope=p['scope']));write(R/'experiment-status.json',{'phase':'completed','updated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed_learning_steps':128,'completed_native_optimizer_steps':256,'success_rule_met':passed})
if __name__=='__main__':main()
