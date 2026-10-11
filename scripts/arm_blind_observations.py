"""Stable public artifact addresses; preserve original physical tool results separately."""
import copy,os
from pathlib import Path
from round031_guard_agent import ExecutionAwareAgent
class ArtifactAddressMap:
 def __init__(self,roots):
  self.roots=tuple(sorted([(str(Path(v).resolve()).rstrip('/')+'/',k) for k,v in roots.items()],key=lambda x:-len(x[0])));assert len(set(x[0] for x in self.roots))==len(self.roots)
 def public(self,value):
  if isinstance(value,dict):return {k:self.public(v) for k,v in value.items()}
  if isinstance(value,list):return [self.public(v) for v in value]
  if isinstance(value,tuple):return tuple(self.public(v) for v in value)
  if isinstance(value,str):
   for root,kind in self.roots:
    if value.startswith(root):return 'artifact://'+kind+'/'+value[len(root):]
    if value==root[:-1]:return 'artifact://'+kind
  return copy.deepcopy(value)
class ArmBlindAgent(ExecutionAwareAgent):
 def __init__(self,*args,**kwargs):
  self.addresses=ArtifactAddressMap({n:os.environ[k] for k,n in [('PDF_RAW_PATH','raw'),('PDF_TRANSLATED_PATH','translated'),('PDF_FIGURES_PATH','figures')]});super().__init__(*args,**kwargs)
 def run(self,*args,**kwargs):self.artifact_observations=[];return super().run(*args,**kwargs)
 def _dispatch_tool(self,name,args):
  actual=super()._dispatch_tool(name,args);public=self.addresses.public(actual);self.artifact_observations.append({'tool':name,'args':copy.deepcopy(args),'actual_result':copy.deepcopy(actual),'public_result':copy.deepcopy(public)});return public
 def _guard_output(self,result):
  result=super()._guard_output(result);result['artifact_observations']=copy.deepcopy(self.artifact_observations);result['stable_public_artifact_addresses']=True;return result
