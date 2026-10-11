"""Generate minimally hooked native loop and prove removing hooks recovers exact source."""
from pathlib import Path
import os,sys,inspect,textwrap,hashlib,json
os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,'AgenticArxiv');from agents.base_agent import BaseAgent
src=textwrap.dedent(inspect.getsource(BaseAgent.run));modified=src.replace('def run(', 'def guarded_native_loop(',1)
changes=[('messages, extra = self.build_messages(enriched_task, tools_description, history_text)','messages, extra = self.build_messages(self._state_task(enriched_task), tools_description, history_text)'),('log.info(f"Thought: {thought}")','log.info(f"Thought: {thought}")\n\n            if self._guard_veto(thought, action_dict, history, step_timings, llm_ms, iteration, msg_id):\n                continue'),('history.append({"thought": thought, "action": "FINISH", "observation": observation})','history.append({"thought": thought, "action": "FINISH", "observation": observation})\n                self._guard_terminal(thought, observation)'),('history.append({"thought": thought, "action": action_str, "observation": observation})','history.append({"thought": thought, "action": action_str, "observation": observation})\n            self._guard_tool(thought, action_dict, observation)'),('return result','return self._guard_output(result)')]
for old,new in changes:assert modified.count(old)==1,old;modified=modified.replace(old,new)
back=modified
for old,new in reversed(changes):assert back.count(new)==1;back=back.replace(new,old)
assert back.replace('def guarded_native_loop(', 'def run(',1)==src
path=Path('scripts/round031_native_loop.py');assert not path.exists();path.write_text('from agents.base_agent import *\n\n'+modified)
p=Path('artifacts/reproduction/round-031-2026-10-09');p.mkdir(exist_ok=True);(p/'native-loop-integration-audit.json').write_text(json.dumps({'native_run_exactly_recovered_after_removing_five_hooks':True,'original_native_source_sha256':hashlib.sha256(Path(inspect.getfile(BaseAgent)).read_bytes()).hexdigest(),'hooked_loop_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'hooks':['state prompt','transaction veto/continue consumesoneiteration','terminal observation','tool observation','output proposal/execution dual traces'],'native_files_unchanged':True},indent=2)+'\n')
