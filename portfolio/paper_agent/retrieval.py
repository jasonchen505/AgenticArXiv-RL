"""Public-query clause retrieval. No fixtures, answers or source paragraph IDs."""
import re,math,collections,copy
from .evidence import PaperEvidenceEnv,terms

def focused_queries(task,pid,catalog):
 # Ownership comes from public names/IDs and natural ordinal references only.
 def mentions(s):
  found=[]
  for c in catalog:
   for label in [c['alias'],c['paper_id']]:
    for m in re.finditer(re.escape(label),s,re.I):found.append((m.start(),c['paper_id']))
  return list(dict.fromkeys(x[1] for x in sorted(found)))
 order=mentions(task);owner=None;out=[]
 for s in re.split(r'[。？！；，!?;\n]',task):
  hit=mentions(s)
  if len(hit)==1:owner=hit[0]
  elif len(hit)>1:owner=None
  elif re.search(r'前一篇|第一篇',s) and order:owner=order[0]
  elif re.search(r'后一篇|第二篇',s) and len(order)>1:owner=order[1]
  if owner==pid and s.strip() and terms(s):out.append(s.strip())
 return list(dict.fromkeys(out)) or [task]

def body_chunks(task,paper,catalog,limit=4):
 cs=[c for c in paper['chunks'] if c['chunk_id'].startswith('body')]
 docs=[collections.Counter(terms(c['text'])) for c in cs];lengths=[sum(d.values()) for d in docs];avg=sum(lengths)/max(1,len(cs));df=collections.Counter(t for d in docs for t in d)
 queries=focused_queries(task,paper['paper_id'],catalog);scores=[]
 for query in queries:
  row=[]
  for d,L in zip(docs,lengths):
   v=0.
   for t in set(terms(query)):
    f=d[t]
    if f:v+=math.log(1+(len(cs)-df[t]+.5)/(df[t]+.5))*f*2.2/(f+1.2*(.25+.75*L/max(1,avg)))
   row.append(v)
  peak=max(row,default=0);scores.append([v/peak if peak else 0 for v in row])
 # Greedily cover distinct query clauses instead of rewarding repeated background.
 selected=[];covered=[0.]*len(scores)
 while len(selected)<min(limit,len(cs)):
  candidates=[i for i in range(len(cs)) if i not in selected]
  def key(i):return (sum(max(0,row[i]-covered[j]) for j,row in enumerate(scores)),sum(row[i] for row in scores),cs[i]['chunk_id'])
  k=max(candidates,key=key)
  # Once every clause has a best match, use the remaining public context quota
  # for the next strongest matches; a best lexical match need not answer it.
  if key(k)[0]<=0:
   k=max(candidates,key=lambda i:(sum(row[i] for row in scores),cs[i]['chunk_id']))
   if sum(row[k] for row in scores)<=0:break
  selected.append(k);covered=[max(v,row[k]) for v,row in zip(covered,scores)]
 return [cs[i] for i in selected],queries

class BodyEvidenceEnv(PaperEvidenceEnv):
 def __init__(self,bank,fail_first_read=False):
  super().__init__(bank,fail_first_read);self.catalog=[{k:p[k] for k in ['paper_id','alias']} for p in bank]
 def _execute(self,name,args):
  result=super()._execute(name,args)
  if name not in {'read_paper','read_cached_paper'} or 'error' in result:return result
  pid=args['paper_id'];paper=self.bank[pid];body,queries=body_chunks(args.get('query',''),paper,self.catalog)
  chunks=[c for c in paper['chunks'] if c['chunk_id'].startswith('abstract')]+body
  self.observed={k:v for k,v in self.observed.items() if v['paper_id']!=pid};ps=[]
  for c in chunks:
   cid=pid+'#'+c['chunk_id'];z={'citation_id':cid,**c,'source_HTML_sha256':paper['source_HTML_sha256'],'paper_id':pid,'title':paper['title']};self.observed[cid]=copy.deepcopy(z);ps.append(z)
  return {'paper_id':pid,'channel':result['channel'],'paragraphs':ps,'retrieval_policy':'public_clause_BM25_coverage_top4','public_queries':queries,'body_chunk_ids':[c['chunk_id'] for c in body]}
