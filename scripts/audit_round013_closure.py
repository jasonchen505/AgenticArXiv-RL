#!/usr/bin/env python3
"""Close the fixed synthetic-negative objective branch, preserve all outcomes."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent
ROOT=HERE if (HERE/'round-012-2026-10-04/completion.json').exists() else HERE.parent/'artifacts/reproduction'
R=ROOT/'round-012-2026-10-04';N=ROOT/'round-013-2026-10-05';N.mkdir(exist_ok=True)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
out=read(R/'mainline-outcome.json');done=read(R/'completion.json')
assert done['actual_new_benchmark_runs']==642 and done['actual_reused_development_runs']==450 and done['actual_new_optimizer_steps']==102
assert out['stop_synthetic_negative_objective_branch'] and not out['both_seeds_beat_continued_sft']
signatures={}
for seed in [43,44]:
    counts={'same_action_and_thought':0,'compared_trials':0,'same_action':0}
    for group in ['dev','previous-iid_test','previous-ood_test','fresh-iid_test','fresh-ood_test']:
        def leaf(a):return a+'-dev' if group=='dev' else group.split('-')[0]+'-'+a+'-'+group.split('-',1)[1]
        docs={a:{(x['task_id'],x['trial']):x for x in [json.loads(s) for s in (R/leaf(a)/'traces.jsonl').read_text().splitlines()]} for a in ['dpo'+str(seed),'anchor'+str(seed)]}
        a,b=docs.values();assert set(a)==set(b)
        def actions(x):
            ys=[]
            for h in x['history']:
                v=h['action']
                try:v=json.loads(v)
                except (json.JSONDecodeError,TypeError):pass
                ys.append(v)
            return ys
        for key in a:
            counts['compared_trials']+=1;counts['same_action']+=actions(a[key])==actions(b[key])
            counts['same_action_and_thought']+=actions(a[key])==actions(b[key]) and [h['thought'].strip() for h in a[key]['history']]==[h['thought'].strip() for h in b[key]['history']]
    assert counts['compared_trials']==156;signatures[str(seed)]=counts
audit={'purpose':'Complete Round012 result closure; no new coefficient search or automatic model promotion.','completion':done,'outcome':out,'training':read(R/'training-audit.json'),'same_seed_action_signatures':signatures,'decisions':{'synthetic_negative_coefficient_branch':'closed','source_sft':'Retain original source and baseline; no exact-author-reproduction claim.','continued_sft':'Retain both seeds as the strongest measured supervised controls; repeated known-world parameter gains coexist with oldDev regression.','next':'Real train-only model-error collection and semantic negative-validity audit; no raw strict failure automatically admitted to training.'},'source_sha256':{f:sha(R/f) for f in ['mainline-outcome.json','completion.json','training-audit.json','freeze-chronology-audit.json','task-level-audit.json','behavior-audit.json','final-source-manifest.json']}}
(N/'round012-closure-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n');print(json.dumps(signatures));print(N/'round012-closure-audit.json')
