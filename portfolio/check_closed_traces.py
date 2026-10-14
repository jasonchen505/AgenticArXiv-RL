"""CPU source/event/prompt parity for a completed evidence study; no generation."""
import argparse,json,hashlib
from pathlib import Path
from portfolio.paper_agent.retrieval import BodyEvidenceEnv
from portfolio.paper_agent.protocol import state_actions,canonical_units,planning_prompt,selection_prompt,parse_plan,parse_selection,render,status_from_events
from portfolio.paper_agent.identity import resolve
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assets',type=Path,required=True);args=p.parse_args();R=args.assets;read=lambda n:json.loads((R/n).read_text());plan=read('experiment-plan.json');bank=read('paper-bank.json');cat=read('public-catalog.json');ts=[json.loads(l) for l in (R/'combined-traces.jsonl').read_text().splitlines()]
 for t in ts:
  env=BodyEvidenceEnv(bank,t['task_id'] in plan['fault_ids'])
  for e in t['events']:
   acts,_,_=state_actions(t['plan'],env,cat,t['task']);assert acts[0]=={'name':e['name'],'args':e['args']};assert env.execute(e['name'],e['args'])==e['result']
  assert env.observed==t['observed_evidence'] and canonical_units(env,t['plan'] or {'paper_ids':[]})==t['units']
  if t['LLM_calls']==2:
   assert t['calls'][0]['request']['messages'][0]['content']==planning_prompt(t['task'],cat)
   if t['plan']:assert parse_plan(t['plan_raw'],cat)==t['plan']
  else:assert resolve(t['task'],cat)==(t['plan'],t['identity_certificate'])
  assert t['calls'][-1]['request']['messages'][0]['content']==selection_prompt(t['task'],t['units'])
  for u in t['units']:
   c=env.observed[u['citation_id']];assert c['text'][u['start']:u['end']]==u['text'];assert hashlib.sha256(u['text'].encode()).hexdigest()==u['sentence_sha256']
  if t['selection'] is not None:
   assert parse_selection(t['answer_raw'],t['units'])==t['selection'];text=render(t['selection'],t['units'],env);prefix='取证执行记录（来自实际工具事件，不是论文主张）：';answer='\n\n'.join(prefix+status_from_events(env.events) if part.startswith(prefix) else part for part in text.split('\n\n'));assert answer==t['answer']
  else:
   assert t['errors'] and not t['answer'],'Rejected outputs never count as delivered answers'
 print(json.dumps({'passed':True,'CPU_only':True,'trajectories':len(ts),'actual_calls_replayed':sum(t['LLM_calls'] for t in ts),'no_model_generations':True,'portable_extraction_semantics_match_closed_study':True}))
if __name__=='__main__':main()
