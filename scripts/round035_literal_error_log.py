"""Preserve the original tool-error Observation while logging braces literally."""
import agents.base_agent as base
class LiteralToolErrorLog:
 def __init__(self,original):self.original=original;self.records=[]
 def __getattr__(self,name):return getattr(self.original,name)
 def error(self,message,*args,**kwargs):
  if not args and kwargs.get('exc_info') is True:
   self.records.append(str(message));return self.original.error('{}',message,**kwargs)
  return self.original.error(message,*args,**kwargs)
def install():
 if isinstance(base.log,LiteralToolErrorLog):return base.log
 base.log=LiteralToolErrorLog(base.log);return base.log
