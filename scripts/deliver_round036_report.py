import json,hashlib,tarfile
from pathlib import Path
ROOT=Path('artifacts/reproduction');ARCHIVE=ROOT/'round036-delivery.tar.gz'
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 with tarfile.open(ARCHIVE,'r:gz') as tar:
  blobs={x.name:tar.extractfile(x).read() for x in tar.getmembers() if x.isfile()};mf='round-036-2026-10-11/delivery-manifest.json';m=json.loads(blobs[mf]);assert set(blobs)==set(m['sha256'])|{mf}
  forbidden={'pilot-status.json','pilot-controller.log','pilot.lock','operation-status.json','operation-controller.log','pilot-outcome.json'}
  for name,b in blobs.items():
   p=Path(name);assert not p.is_absolute() and '..' not in p.parts and p.parts[0]=='round-036-2026-10-11';dest=ROOT/p
   if len(p.parts)==2:assert p.name not in forbidden and (not p.name.endswith('-job.json') or p.name.startswith('snapshot-')) and (not p.name.endswith('.log') or p.name.startswith('snapshot-') or p.name=='curriculum-CPU.log'),'Active originals excluded'
   if name!=mf:assert sha(b)==m['sha256'][name]
   if dest.exists():assert sha(dest.read_bytes())==sha(b),'Refusing overwrite: '+str(dest)
  for n,h in m['historical_HTML_sha256'].items():assert sha((ROOT/n).read_bytes())==h
  for name,b in blobs.items():
   dest=ROOT/name
   if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
  for name,h in m['sha256'].items():assert sha((ROOT/name).read_bytes())==h
  print(json.dumps({'verified_static_files':len(m['sha256']),'report_SHA256':m['report_sha256'],'old_reports_preserved':True,'active_jobs_not_modified':True}))
if __name__=='__main__':main()
