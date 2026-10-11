#!/usr/bin/env python3
"""Use existing BenchmarkRunner with a hash-frozen task JSON; no learner edits."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

def main():
    p=argparse.ArgumentParser()
    for name in ['tasks','split','model','snapshot','output','tasks_sha256']:p.add_argument('--'+name,required=True)
    p.add_argument('--prefix',default='bench_r20261003_075529')
    p.add_argument('--seed',type=int,default=45);p.add_argument('--repeat',type=int,default=3)
    a=p.parse_args();file=Path(a.tasks)
    assert hashlib.sha256(file.read_bytes()).hexdigest()==a.tasks_sha256
    tasks=json.loads(file.read_text());path,group=a.split.rsplit(':',1)
    wanted=json.loads(Path(path).read_text())['split'][group]
    byid={t['id']:t for t in tasks};assert len(byid)==len(tasks)
    tasks=[byid[t] for t in sorted(wanted)]
    out=Path(a.output);assert not out.exists();out.mkdir(parents=True)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'AgenticArxiv'))
    from tools.bootstrap import require_all_tools
    from benchmark.runner import BenchmarkRunner
    from benchmark.report import BenchmarkReport
    require_all_tools('Frozen task file benchmark')
    runner=BenchmarkRunner(agent_types=['regex'],repeat=a.repeat,model=a.model,session_prefix=a.prefix,offline=True,snapshot=a.snapshot,llm_backend='transformers',local_device='cuda',local_dtype='bfloat16',generation_seed=a.seed,prompt_policy='baseline')
    results=runner.run_all(tasks)
    traces=[{'task_id':r.task_id,'agent_type':r.agent_type,'trial':r.trial,'session_id':r.session_id,'history':(r.raw_result or {}).get('history')} for r in results]
    (out/'traces.jsonl').write_text(''.join(json.dumps(t,ensure_ascii=False)+'\n' for t in traces))
    errors=[{'session_id':r.session_id,'task_id':r.task_id,'agent_type':r.agent_type,'trial':r.trial,'error':r.error} for r in results if r.error]
    metrics=[r.metrics for r in results if r.metrics is not None]
    report=BenchmarkReport(metrics,model=a.model,errors=errors);report.save_all(str(out))
    protocol={'prompt_policy':'baseline','model':a.model,'snapshot':a.snapshot,'split':a.split,'tasks':a.tasks,'tasks_sha256':a.tasks_sha256,'repeat':a.repeat,'seed':a.seed,'session_prefix':a.prefix,'backend':'transformers','dtype':'bfloat16','weights_updated':False,'oracle_in_prompt':False,'task_loader':'Existing BenchmarkRunner.run_all with frozen task definitions'}
    (out/'execution_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    assert len(metrics)==len(tasks)*a.repeat and not errors
    print('Completed',len(metrics),'frozen-task runs',flush=True)

if __name__=='__main__':main()
