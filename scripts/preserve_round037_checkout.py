from pathlib import Path
import os,subprocess,json,datetime,hashlib,tarfile
ROOT=Path('/data/home/chenyizhou/AgenticArXiv-RL');R=ROOT/'artifacts/reproduction/round-037-2026-10-11';F=ROOT.parent/'AgenticArXiv-RL-repro-frozen-20261011'
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,x):(R/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 assert json.loads((ROOT/'artifacts/reproduction/round-036-2026-10-11/operation-status.json').read_text())['phase']=='completed';assert not (ROOT/'.git/MERGE_HEAD').exists();assert not F.exists();R.mkdir(exist_ok=True);assert not (R/'preservation.json').exists();base=git('rev-parse','HEAD');upstream=git('rev-parse','origin/main');tracked=git('diff','--name-only').splitlines();untracked=git('ls-files','--others','--exclude-standard').splitlines();code=[x for x in untracked if x.startswith(('scripts/','AgenticArxiv/tests/','docs/'))];selected=tracked+code;assert len(tracked)==14;assert not set(untracked).intersection(git('ls-tree','-r','--name-only','origin/main').splitlines()),'Upstream/untracked collision';hashes={n:sha(ROOT/n) for n in selected};(R/'local-before-merge.patch').write_bytes(subprocess.check_output(['git','diff','--binary'],cwd=ROOT));(R/'upstream-commits.txt').write_text(git('log','--reverse','--oneline',base+'..'+upstream)+'\n');(R/'upstream-stat.txt').write_text(git('diff','--stat',base+'..'+upstream)+'\n')
 with tarfile.open(R/'pre-sync-code.tar.gz','w:gz') as t:
  for n in selected:t.add(ROOT/n,arcname=n,recursive=False)
 write('before-sync.json',{'recorded_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'branch':git('branch','--show-current'),'base':base,'upstream':upstream,'tracked_dirty_files':tracked,'untracked_code_files':code,'excluded_artifacts_and_data_remain_on_disk':True,'source_SHA256':hashes,'backup_archive_SHA256':sha(R/'pre-sync-code.tar.gz')})
 subprocess.run(['git','add','--']+selected,cwd=ROOT,check=True);subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@users.noreply.github.com','commit','-m','chore: preserve reproduction code before upstream synchronization'],cwd=ROOT,check=True,stdout=subprocess.PIPE,text=True);snapshot=git('rev-parse','HEAD');subprocess.run(['git','worktree','add','-b','codex/repro-frozen-20261011',str(F),snapshot],cwd=ROOT,check=True,stdout=subprocess.PIPE,text=True)
 # Keep old code independent; large existing datasets, weights and evidence are shared read-only for audit.
 links={}
 for name in ['.venv-repro','outputs']:
  src=ROOT/name;dest=F/name
  if src.exists() and not dest.exists():dest.symlink_to(src,target_is_directory=src.is_dir());links[str(dest)]=str(src)
 for parent in ['data','artifacts']:
  (F/parent).mkdir(exist_ok=True)
  for src in (ROOT/parent).iterdir():
   dest=F/parent/src.name
   if not dest.exists():dest.symlink_to(src,target_is_directory=src.is_dir());links[str(dest)]=str(src)
 for n,h in hashes.items():assert sha(F/n)==h,n
 write('preservation.json',{'recorded_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'base':base,'snapshot_commit':snapshot,'upstream_target':upstream,'frozen_checkout':str(F),'frozen_branch':'codex/repro-frozen-20261011','code_files_SHA_verified':len(hashes),'shared_data_weight_evidence_links':links,'large_artifacts_not_committed':True,'local_commit_only_no_push':True});print(json.loads((R/'preservation.json').read_text()))
if __name__=='__main__':main()
