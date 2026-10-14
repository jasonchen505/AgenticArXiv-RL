import json,re,hashlib,types
def unique(pairs):
 d={}
 for k,v in pairs:
  if k in d:raise ValueError('duplicate key')
  d[k]=v
 return d
def planning_prompt(task,catalog):
 schema={'type':'object','properties':{'paper_ids':{'type':'array','minItems':1,'maxItems':3,'uniqueItems':True,'items':{'type':'string','enum':[p['paper_id'] for p in catalog]}}},'required':['paper_ids'],'additionalProperties':False}
 return ('Select only the papers explicitly requested by the user from the PUBLIC metadata catalog. Datasets, benchmarks, models and failure-handling conditions are not extra papers. '
 'Output one JSON object with the single key paper_ids. Do not answer or decompose the request. Do not add papers about vaguely related topics. '
 'Output schema: '+json.dumps(schema)+'\nPUBLIC metadata catalog:\n'+'\n'.join(p['paper_id']+' | '+p['alias']+' | '+p['title'] for p in catalog)+'\nOriginal user request:\n'+task)

def parse_plan(raw,catalog):
 x=json.loads(raw,object_pairs_hook=unique);known={p['paper_id'] for p in catalog}
 if not isinstance(x,dict) or set(x)!={'paper_ids'} or not isinstance(x['paper_ids'],list) or not 1<=len(x['paper_ids'])<=3:raise ValueError('paper-only plan fields')
 if any(not isinstance(pid,str) or pid not in known for pid in x['paper_ids']) or len(set(x['paper_ids']))!=len(x['paper_ids']):raise ValueError('public catalog IDs')
 return x
def units(env):
 us=[]
 for cid,c in env.observed.items():
  text=c['text'];starts=[0]+[m.end() for m in re.finditer(r'(?<=[.!?])\s+(?=[A-Z(])',text)];ends=starts[1:]+[len(text)]
  for start,end in zip(starts,ends):
   sentence=text[start:end].rstrip();end=start+len(sentence)
   us.append({'id':len(us),'citation_id':cid,'paper_id':c['paper_id'],'title':c['title'],'text':sentence,'start':start,'end':end,'sentence_sha256':hashlib.sha256(sentence.encode()).hexdigest(),'paragraph_sha256':c['sha256'],'source_HTML_sha256':c['source_HTML_sha256'],'source_url':c['source_url']})
 return us

def selection_prompt(task,us):
 schema={'type':'object','properties':{'sentence_ids':{'type':'array','minItems':0,'maxItems':8,'uniqueItems':True,'items':{'type':'integer','minimum':0,'maximum':len(us)-1}}},'required':['sentence_ids'],'additionalProperties':False}
 return ('Select the smallest set of ACTUALLY OBSERVED sentences that directly answers every paper-fact part of the ORIGINAL user request. '
 'Return exactly one JSON object {"sentence_ids":[integer IDs]}. No prose, Markdown, extra keys or fabricated sentences. '
 'Keep paper identities correct. Choose evidence of the actual requested methods or results, not unrelated background or benchmarks. '
 'Source outage/cache recovery is our execution status, not a paper fact. The renderer will independently display the real tool receipt. '
 'No measurements of this project were provided, so paper sentences cannot establish our completion-rate or model-call savings; the renderer will separately state this measurement limit. '
 'Output schema: '+json.dumps(schema)+'\nORIGINAL user request:\n'+task+'\nACTUAL sentence pool:\n'+json.dumps([{'id':u['id'],'paper_id':u['paper_id'],'sentence':u['text']} for u in us],ensure_ascii=False))

def parse_selection(raw,us):
 x=json.loads(raw,object_pairs_hook=unique)
 if not isinstance(x,dict) or set(x)!={'sentence_ids'} or not isinstance(x['sentence_ids'],list) or len(x['sentence_ids'])>8:raise ValueError('sentence selection fields')
 ids=x['sentence_ids']
 if any(type(i)!=int or not 0<=i<len(us) for i in ids) or len(set(ids))!=len(ids):raise ValueError('sentence IDs; no normalization')
 return x

def render(x,us,env):
 sections=[]
 for pid in dict.fromkeys(us[i]['paper_id'] for i in x['sentence_ids']):
  selected=[us[i] for i in x['sentence_ids'] if us[i]['paper_id']==pid];sections.append(selected[0]['title']+'\n'+'\n'.join('> '+u['text']+' ['+u['citation_id']+']('+u['source_url']+')' for u in selected))
 if not sections:sections=['本次实际观察没有被模型选为充分证据；不意味着整篇论文没有相关内容。']
 failures=[e for e in env.events if not e['success']];recoveries=[e for e in env.events if e['name']=='read_cached_paper' and e['success']]
 sections.append('取证执行记录（来自实际工具事件，不是论文主张）：'+('主源实际失败；执行器按反馈读取同版本缓存，恢复成功。' if env.fired and recoveries else '没有完成缓存恢复。' if failures else '主源读取成功。'))
 sections.append('本项目测量边界（来自输入/运行记录，不是论文结论）：本次没有提供我们Agent的任务完成率或模型调用对照测量；不能仅凭论文断定本项目保持完成率且减少模型调用。')
 return '\n\n'.join(sections)
def state_actions(p,env,catalog,task):
 byid={c['paper_id']:c for c in catalog};arrived={c['paper_id'] for c in env.observed.values()};actions=[];identity_failed=[]
 for pid in p['paper_ids']:
  if pid in arrived:continue
  matches=[e for e in env.events if e['name']=='search_papers' and e['args'].get('query')==byid[pid]['title']]
  if not matches:
   actions.append({'name':'search_papers','args':{'query':byid[pid]['title'],'max_results':1}});continue
  e=matches[-1]
  if not e['success'] or not isinstance(e['result'],list) or not e['result'] or e['result'][0]['paper_id']!=pid:identity_failed.append(pid);continue
  if pid in env.failed:actions.append({'name':'read_cached_paper','args':{'paper_id':pid,'query':task}})
  else:actions.append({'name':'read_paper','args':{'paper_id':pid,'query':task}})
 return actions,arrived,identity_failed

def canonical_units(env,p):
 order={pid:i for i,pid in enumerate(p['paper_ids'])}
 cs=sorted(env.observed.items(),key=lambda kv:(order.get(kv[1]['paper_id'],999),0 if kv[1]['chunk_id'].startswith('abstract') else 1,kv[1]['chunk_id']))
 return units(types.SimpleNamespace(observed=dict(cs)))
def status_from_events(events):
 reads=[e for e in events if e['name'] in ['read_paper','read_cached_paper']]
 primary=[e for e in reads if e['name']=='read_paper']
 failed=[e for e in primary if not e['success']]
 if failed:
  cached={e['args']['paper_id'] for e in reads if e['name']=='read_cached_paper' and e['success']}
  pending={e['args']['paper_id'] for e in failed}-cached
  if pending:return '主源实际失败；尚未完成全部失败论文的同版本缓存恢复。'
  return '主源实际失败；执行器按反馈读取同版本缓存，恢复成功。'
 if not primary:
  if any(e['name']=='read_cached_paper' and e['success'] for e in reads):return '实际读取了同版本缓存；未执行主源读取。'
  return '未执行主源读取；不能报告读取成功。'
 if all(e['success'] for e in primary):return '主源读取成功。'
 raise AssertionError('Unclassified actual read state')
