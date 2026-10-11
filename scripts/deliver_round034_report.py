import json,hashlib,tarfile
from pathlib import Path
ROOT=Path('artifacts/reproduction');ARCHIVE=ROOT/'round034-delivery.tar.gz'
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 with tarfile.open(ARCHIVE,'r:gz') as tar:
  blobs={x.name:tar.extractfile(x).read() for x in tar.getmembers() if x.isfile()};mf='round-034-2026-10-10/delivery-manifest.json';m=json.loads(blobs[mf]);assert set(blobs)==set(m['sha256'])|{mf}
  forbidden={'experiment-status.json','controller.log','experiment.lock','all-finals-before-inference.json','study-outcome.json'}
  for name,b in blobs.items():
   p=Path(name);assert not p.is_absolute() and '..' not in p.parts and p.parts[0]=='round-034-2026-10-10';dest=ROOT/p
   if len(p.parts)==2:assert p.name not in forbidden and (not p.name.endswith('-train-job.json') or p.name.startswith('snapshot-')) and (not p.name.endswith('-train.log') or p.name.startswith('snapshot-')) and (not p.name.endswith('-job.json') or p.name.startswith(('preflight','checkpointed-preflight','snapshot-'))),'Active originals excluded'
   if name!=mf:assert sha(b)==m['sha256'][name]
   if dest.exists():assert sha(dest.read_bytes())==sha(b),'Refusing to overwrite existing evidence: '+str(dest)
  assert sha((ROOT/'round-033-2026-10-10/report.html').read_bytes())==m['historical_R033_HTML_SHA']
  for name,b in blobs.items():
   dest=ROOT/name
   if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
  for name,h in m['sha256'].items():assert sha((ROOT/name).read_bytes())==h
  print(json.dumps({'verified_static_files':len(m['sha256']),'report_SHA256':m['report_sha256'],'old_R033_HTML_preserved':True,'active_jobs_not_modified':True}))
if __name__=='__main__':main()
