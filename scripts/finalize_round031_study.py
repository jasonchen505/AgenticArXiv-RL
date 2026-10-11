import os,datetime
os.environ['CUDA_VISIBLE_DEVICES']=''
from round031_study_core import R,read,write,sha,collect,success_checks,compare

def main():
 assert not (R/'study-outcome.json').exists();s=collect(full=True);p=read(R/'experiment-plan.json');files=[]
 for seed in p['generation_seeds']:
  for group in p['groups']:
   for before,after in p['comparison_pairs']:
    v=compare(s,seed,group,before,after);f=R/(group+'-'+before+'-vs-'+after+'-s'+str(seed)+'-comparison.json');assert not f.exists();write(f,v);files.append(str(f))
 assert len(files)==48;checks=success_checks(s['counts']);met=all(checks.values());write(R/'study-outcome.json',dict(s,finalized_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=sha(R/'experiment-plan.json'),comparisons=files,rule_checks=checks,registered_rule_met=met,fixed_recipe='passed_requiring_review' if met else 'closed',no_automatic_promotion=True,source_sft_remains_default=True,learning_steps=0));print({'full_trials':2112,'comparisons':48,'rules':checks,'passed':met})
if __name__=='__main__':main()
