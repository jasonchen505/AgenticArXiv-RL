#!/usr/bin/env python3
from pathlib import Path
import datetime,json,subprocess
from audit_round027_status import collect,read,sha,R
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 assert not (R/'retention-outcome.json').exists();s=collect(full=True);plan=read(R/'experiment-plan.json');c=s['counts'];comparisons=[]
 for group,g in plan['groups'].items():
  sf,key=g['split'].rsplit(':',1)
  for seed in [43,44]:
   suffix=str(seed);pairs=[('klrepeat'+suffix,'klcurriculum'+suffix),('plaincurriculum'+suffix,'klcurriculum'+suffix),('plainrepeat'+suffix,'klrepeat'+suffix),('plainrepeat'+suffix,'plaincurriculum'+suffix),('sft','klcurriculum'+suffix)]
   for base,target in pairs:
    d0=plan['reused_evals'].get(group,{}).get(base,str(R/(group+'-'+base)));d1=plan['reused_evals'].get(group,{}).get(target,str(R/(group+'-'+target)));out=R/(base+'-vs-'+target+'-'+group+'.json')
    subprocess.run(['.venv-repro/bin/python','scripts/compare_repro_evals.py','--base',d0,'--sft',d1,'--oracles',g['oracles'],'--output',str(out),'--base_label',base,'--sft_label',target,'--training_description','Fixed0.1completion-forwardKL/originalTRAIN anchors factorial data study;historical plainCE same795rows199steps but fewerforwards;notexactcompute budget;terminal semanticmode nuisance disclosed','--evaluation_split',key],check=True)
    assert all(x['authoritative_rescore_drift']==0 for x in read(out)['models'].values());comparisons.append(str(out))
 def each(predicate):return all(predicate(str(seed)) for seed in [43,44])
 checks={'each_seed_each_fresh_world_course_gain_at_least3_vs_KL_repeat':each(lambda s:all(c[w]['klcurriculum'+s]['strict']>=c[w]['klrepeat'+s]['strict']+3 for w in ['freshA','freshB'])),'oldDev_strict_at_least_source_and_KL_repeat':each(lambda s:c['olddev']['klcurriculum'+s]['strict']>=max(10,c['olddev']['klrepeat'+s]['strict'])),'oldDev_FF_no_worse_than_source_and_KL_repeat':each(lambda s:c['olddev']['klcurriculum'+s]['false_finish']<=min(6,c['olddev']['klrepeat'+s]['false_finish'])),'newDev_strict_retained_vs_plain_course_and_KL_repeat':each(lambda s:c['newdev']['klcurriculum'+s]['strict']>=max(c['newdev']['plaincurriculum'+s]['strict'],c['newdev']['klrepeat'+s]['strict'])),'newDev_FF_no_worse_than_plain_course_and_KL_repeat':each(lambda s:c['newdev']['klcurriculum'+s]['false_finish']<=min(c['newdev']['plaincurriculum'+s]['false_finish'],c['newdev']['klrepeat'+s]['false_finish'])),'fresh_strict_retained_vs_plain_course':each(lambda s:all(c[w]['klcurriculum'+s]['strict']>=c[w]['plaincurriculum'+s]['strict'] for w in ['freshA','freshB'])),'fresh_FF_no_worse_than_KL_repeat':each(lambda s:all(c[w]['klcurriculum'+s]['false_finish']<=c[w]['klrepeat'+s]['false_finish'] for w in ['freshA','freshB']))}
 assert set(checks)==set(plan['success_rule']) and len(comparisons)==40;success=all(checks.values())
 write(R/'retention-outcome.json',dict(s,completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),comparisons=comparisons,rule_checks=checks,registered_rule_met=success,fixed_recipe='completed_keep_unpromoted' if success else 'closed',no_automatic_promotion=True,scope=plan['scope']))
 write(R/'experiment-status.json',{'phase':'completed','updated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed_training_steps':796,'success_rule_met':success})
if __name__=='__main__':main()
