from pathlib import Path
import json,tarfile
b=Path('artifacts/reproduction');r=b/'round-030-2026-10-09';a=json.loads((r/'closure-independent-audit.json').read_text());files=set(p for p in r.rglob('*') if p.is_file())
for p in a['evidence_sha256']:files.add(Path(p))
for p in a['comparisons']:files.add(Path(p))
old=b/'round-029-2026-10-09'
for n in ['experiment-plan.json','all-finals-before-inference.json','online-outcome.json','experiment-status.json']:files.add(old/n)
for arm in ['rl43','lr0_43','rl44','lr0_44']:
 for n in ['rollout_traces.jsonl','final/training_manifest.json']:
  src=Path('outputs/repro_round029')/arm/n;dest=r/(arm+'-'+('rollout-traces.jsonl' if 'rollout' in n else 'training-manifest.json'));dest.write_bytes(src.read_bytes());files.add(dest)
for p in old.glob('*-job.json'):files.add(p)
with tarfile.open('artifacts/reproduction/round030-evidence.tar.gz','w:gz') as t:
 for p in sorted(files):t.add(p,arcname=str(p.relative_to(b)))
print('staticfiles',len(files))
