from pathlib import Path
import json,tarfile,datetime,subprocess
from round031_study_core import read,write,sha,R
N=R.parent/'round-032-2026-10-10';a=read(N/'closure-independent-audit.json');files={Path(f) for f in a['evidence_sha256']};files.update(f for f in N.iterdir() if f.is_file() and not f.name.endswith('.log') and not f.name.endswith('.lock') and not f.name.endswith('-job.json'));files.update(R/f for f in ['experiment-status.json','CPU-completion.log','controller.log','original-status-before-CPU-completion.json','report.html','report-qa.json','launch.json','source-model-before-inference.json','CPU-completion-launch.json']);p=read(N/'parity-plan.json');status=read(N/'parity-status.json') if (N/'parity-status.json').exists() else {};jobs={}
for f in N.glob('parity-*-job.json'):
 j=read(f);assert j['gpu_index'] in [4,7];jobs[f.name]=j;dest=N/('snapshot-'+f.name);write(dest,j);files.add(dest)
mode=status.get('current_job');log=N/('parity-'+str(mode)+'.log');lines=log.read_text(errors='replace').splitlines() if log.exists() else [];launch=read(N/'parity-launch.json');ps=subprocess.run(['ps','-p',str(launch['pid']),'-o','pid,etime,args'],capture_output=True,text=True);snap={'captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':status,'launch':launch,'controller_alive':ps.returncode==0,'controller_ps':ps.stdout.strip(),'jobs':jobs,'task_starts':sum(x.startswith('[') and '] task=' in x for x in lines),'log_tail':lines[-20:],'new_model_efficacy_claim':False,'learning_steps':0,'source_unchanged':True};write(N/'parity-at-report.json',snap);files.add(N/'parity-at-report.json')
if log.exists():dest=N/('snapshot-'+log.name);dest.write_text(log.read_text(errors='replace'));files.add(dest)
for f in N.glob('observation-CPU*.log'):files.add(f)
with tarfile.open(N.parent/'round032-evidence.tar.gz','w:gz') as t:
 for f in sorted(files):t.add(f,arcname=str(f.relative_to(N.parent)))
print({'static_files':len(files),'archive_sha256':sha(N.parent/'round032-evidence.tar.gz'),'task_starts':snap['task_starts'],'status':status})
