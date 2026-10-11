from pathlib import Path
import json,tarfile
from round029_online_core import R,collect,write
assert not (R/'learning-at-report.json').exists();s=collect();write(R/'learning-at-report.json',s);arm=s['status'].get('current_job','rl43');(R/'learning-log-at-report.txt').write_bytes((R/(arm+'-train.log')).read_bytes());job=R/(arm+'-train-job.json');write(R/'learning-job-at-report.json',json.loads(job.read_text()))
b=R.parent;files=set(p for p in R.iterdir() if p.is_file() and p.name not in ['experiment-status.json','controller.log','experiment.lock'] and not p.name.endswith('-train-job.json') and not p.name.endswith('-train.log'))
for p in s['evidence_sha256']:files.add(Path(p))
for n in ['program-probe-outcome.json','program-probe-job.json','probe-status.json']:files.add(b/'round-028-2026-10-09'/n)
for n in ['rollout_traces.jsonl','rollout_traces.summary.json','reward_probe_summary.json']:
 p=R/('closed-probe-'+n);p.write_bytes((Path('outputs/repro_round028/program_probe')/n).read_bytes());files.add(p)
p=R/'closed-probe-training-manifest.json';p.write_bytes(Path('outputs/repro_round028/program_probe/final/training_manifest.json').read_bytes());files.add(p)
with tarfile.open('/tmp/round029-evidence.tar.gz','w:gz') as t:
 for p in sorted(files):t.add(p,arcname=str(p.relative_to(b)))
print('staticfiles',len(files),'snapshot',s['logged_native_steps'],s['captured_at_utc'])
