#!/usr/bin/env python3
"""Train-only replay: inspect whether a terminal cap changes GRPO advantages."""
import collections, hashlib, json, math, sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'AgenticArxiv'))
from benchmark.metrics import extract_metrics,is_strict_success
R=REPO/'artifacts/reproduction/round-008-2026-10-04'
R.mkdir(parents=True,exist_ok=True)
oracles={x['id']:x for x in json.loads((REPO/'artifacts/reproduction/round-002-2026-10-02/task-oracles.json').read_text())}

def advantage(values):
    mean=sum(values)/len(values)
    std=math.sqrt(sum((v-mean)**2 for v in values)/(len(values)-1))
    return [(v-mean)/(std+1e-4) for v in values]

def main():
    arms={}
    for arm,round_,source in [('hard43','006','hard7'),('broad43','006','broad51'),('hard44','007','hard44'),('broad44','007','broad44')]:
        directory='round-006-2026-10-03' if round_=='006' else 'round-007-2026-10-04'
        file=REPO/'artifacts/reproduction'/directory/(source+'-rollout-traces.jsonl')
        rows=[json.loads(x) for x in file.read_text().splitlines()]
        groups=collections.defaultdict(list);counts=collections.Counter();changes=[]
        for x in rows:
            task=oracles[x['task_id']]
            m=extract_metrics(task,x['trajectory'],'regex',x['generation_index'])
            old=x['reward']; triggered=m.termination_type=='FINISH' and m.terminal_semantics_applicable and not m.terminal_semantics_accurate
            new=min(old,0.0) if triggered else old
            counts['samples']+=1;counts['strict_failures']+=not is_strict_success(m)
            counts['blocked_samples']+=m.terminal_semantics_applicable
            counts['wrong_blocked_reason']+=triggered;counts['reward_changed_samples']+=new!=old
            counts['positive_reward_strict_failures']+=old>0 and not is_strict_success(m)
            groups[(x['training_step'],x['group_index'])].append((x,old,new,triggered))
            if new!=old:changes.append({'task_id':x['task_id'],'training_step':x['training_step'],'generation':x['generation_index'],'old_reward':old,'new_reward':new,'terminal_thought':x['trajectory']['history'][-1].get('thought','')})
        group_changes=[]
        for key,group in groups.items():
            assert len(group)==4
            old=advantage([x[1] for x in group]);new=advantage([x[2] for x in group])
            delta=max(abs(a-b) for a,b in zip(old,new));flips=sum(a>1e-3 and b<-1e-3 for a,b in zip(old,new))
            counts['groups']+=1;counts['material_advantage_changed_groups']+=delta>0.01
            counts['positive_to_negative_advantage_samples']+=flips
            if delta>0.01:group_changes.append({'training_step':key[0],'group_index':key[1],'task_id':group[0][0]['task_id'],'old_rewards':[x[1] for x in group],'new_rewards':[x[2] for x in group],'old_advantages':old,'new_advantages':new,'max_advantage_delta':delta,'positive_to_negative':flips})
        arms[arm]={'counts':dict(counts),'trace_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'reward_changes':changes,'material_group_changes':group_changes}
    result={'data_scope':'Only archived train-only rollouts, no benchmark traces or fresh test examples read.','objective':'blocked_reason_v1: FINISH + expected blocked terminal + incorrect terminal semantics => min(legacy_reward,0). Other rewards unchanged.','advantage_formula':'(reward-group_mean)/(sample_std+1e-4), four rollouts; material change > .01. Replay is diagnostic, not a new training outcome.','arms':arms}
    (R/'train-only-objective-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({a:x['counts'] for a,x in arms.items()},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
