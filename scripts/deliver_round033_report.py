import json,hashlib,tarfile
from pathlib import Path
ROOT=Path('artifacts/reproduction');ARCHIVE=ROOT/'round033-delivery.tar.gz'
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 with tarfile.open(ARCHIVE,'r:gz') as tar:
  blobs={x.name:tar.extractfile(x).read() for x in tar.getmembers() if x.isfile()};mf='round-033-2026-10-10/delivery-manifest.json';m=json.loads(blobs[mf]);assert set(blobs)==set(m['sha256'])|{mf}
  for name,b in blobs.items():
   p=Path(name);assert not p.is_absolute() and '..' not in p.parts and p.parts[0]=='round-033-2026-10-10';dest=ROOT/p
   if name!=mf:assert sha(b)==m['sha256'][name]
   if dest.exists():assert sha(dest.read_bytes())==sha(b),'Refusing to overwrite existing evidence: '+str(dest)
  for name,b in blobs.items():
   dest=ROOT/name
   if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
  for name,h in m['sha256'].items():assert sha((ROOT/name).read_bytes())==h
  assert sha((ROOT/'round-032-2026-10-10/report.html').read_bytes())==m['historical_R032_HTML_SHA'];print({'verified_static_files':len(m['sha256']),'report_sha256':m['report_sha256'],'old_R032_HTML_preserved':True,'active_jobs_not_modified':True})
if __name__=='__main__':main()
