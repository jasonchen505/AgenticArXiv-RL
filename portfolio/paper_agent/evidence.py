import hashlib,json,math,re,time,copy
from pathlib import Path
def terms(text):
 synonyms={'模块':'planner worker solver components','组件':'components planner executor','反思':'reflection overhead','成本':'cost','冗余':'redundant waste','并行':'parallel','效率':'efficiency','结果':'results','输入':'input','开销':'overhead','时延':'latency','轨迹':'trajectory'}
 for a,b in synonyms.items():text=text.replace(a,' '+b+' ')
 return re.findall(r'[a-z0-9]+',text.lower())

def rank(query,text):
 q=set(terms(query));t=terms(text);return sum(min(t.count(x),4) for x in q)/math.sqrt(max(1,len(t)))

class PaperEvidenceEnv:
 def __init__(self,bank,fail_first_read=False):
  self.bank={x['paper_id']:x for x in bank};self.known=set();self.observed={};self.events=[];self.failed=set();self.fail_first_read=fail_first_read;self.fired=False
 def _execute(self,name,args):
  if not isinstance(args,dict):raise ValueError('args必须是object')
  if name=='search_papers':
   if set(args)-{'query','max_results'}:raise ValueError('未知search参数')
   q=args['query'];n=args.get('max_results',3)
   if not isinstance(q,str) or not q.strip() or type(n)!=int or not 1<=n<=3:raise ValueError('非法query/max_results')
   hits=sorted(self.bank.values(),key=lambda p:(rank(q,p['title']+' '+p['alias']),p['paper_id']),reverse=True)
   hits=[p for p in hits if rank(q,p['title']+' '+p['alias'])>0][:n]
   self.known.update(p['paper_id'] for p in hits)
   return [{'paper_id':p['paper_id'],'title':p['title'],'source_url':p['source_url']} for p in hits]
  if name not in {'read_paper','read_cached_paper'}:raise ValueError('工具不存在: '+str(name))
  if set(args)-{'paper_id','query'}:raise ValueError('未知read参数')
  pid=args['paper_id'];q=args.get('query','')
  if pid not in self.known:raise ValueError('paper_id未由搜索得到；请先搜索')
  if not isinstance(q,str):raise ValueError('query必须为string')
  if name=='read_cached_paper' and pid not in self.failed:raise ValueError('主源未失败，备份未开放')
  if name=='read_paper' and self.fail_first_read and not self.fired:
   self.fired=True;self.failed.add(pid)
   return {'error':'主源临时不可用（固定故障注入）','paper_id':pid,'recoverable':True,'available_tool':'read_cached_paper','same_version_cache':True}
  paper=self.bank[pid];abstract=[c for c in paper['chunks'] if c['chunk_id'].startswith('abstract')]
  body=sorted([c for c in paper['chunks'] if c not in abstract],key=lambda c:(rank(q,c['text']),c['chunk_id']),reverse=True)
  selected=abstract+body[:2]
  result=[]
  for c in selected:
   cid=pid+'#'+c['chunk_id'];value={'citation_id':cid,**c,'source_HTML_sha256':paper['source_HTML_sha256'],'paper_id':pid,'title':paper['title']}
   self.observed[cid]=copy.deepcopy(value);result.append(value)
  return {'paper_id':pid,'channel':'cache' if name=='read_cached_paper' else 'primary','paragraphs':result}
 def execute(self,name,args):
  start=time.monotonic()
  try:result=self._execute(name,args)
  except (KeyError,ValueError,TypeError) as e:result={'error':str(e)}
  event={'index':len(self.events)+1,'name':name,'args':copy.deepcopy(args),'result':copy.deepcopy(result),'success':not(isinstance(result,dict) and 'error' in result),'tool_ms':int((time.monotonic()-start)*1000)}
  self.events.append(event);return result
