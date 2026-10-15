"""Audit and browse published historical records using only the standard library.

This module does not run a model, contact paper tools, or rejudge semantic quality.
"""
from pathlib import Path
import argparse,hashlib,html,json,math

ARM_NAMES={'qwen35_on_compiled':'开启思考 / 压缩','qwen35_on_two':'开启思考 / 两调用','qwen35_off_compiled':'关闭思考 / 压缩'}

def require(condition,message):
 if not condition:raise ValueError(message)

def load_bundle(root=None):
 root=Path(root) if root else Path(__file__).parent/'evidence';hashes={}
 def read(name):
  b=(root/name).read_bytes();hashes[name]=hashlib.sha256(b).hexdigest();return json.loads(b)
 bundle={'results':read('results.json'),'outcome':read('round056-outcome.json'),'tasks':read('round056-public-tasks.json'),'catalog':read('round056-public-catalog.json'),'reviews':read('round056-selected-evidence.json')}
 provenance=read('training-records/provenance.json');bundle['training']={}
 for name,ref in provenance['records'].items():
  require(Path(name).name==name,'Unsafe record filename')
  value=read('training-records/'+name);require(hashes['training-records/'+name]==ref['public_SHA256'],'Training record hash mismatch: '+name)
  original=bundle['results']['scientific_evidence_SHA256'].get(ref['source'])
  require(original is None or original==ref['source_SHA256'],'Original record reference mismatch: '+name)
  bundle['training'][name]=value
 return bundle,hashes

def audit_bundle(b):
 o=b['outcome'];require(b['results']['qwen35_body_study']==o,'Outcome differs from published result index')
 tasks={t['id']:t for t in b['tasks']};catalog={p['paper_id']:p for p in b['catalog']}
 require(len(tasks)==len(b['tasks'])==8 and len(catalog)==len(b['catalog'])==6,'Unexpected registered task/catalog set')
 for pid,p in catalog.items():require(p['source_url']=='https://arxiv.org/html/'+pid,'Catalog source binding mismatch')
 reviews={};quote_count=0
 for r in b['reviews']:
  key=(r['arm'],r['task_id']);require(r['arm'] in ARM_NAMES and r['task_id'] in tasks and key not in reviews,'Duplicate or unknown review')
  require(type(r['task_success']) is bool and isinstance(r['missing'],list) and len(r['missing'])<=4,'Invalid semantic review record')
  require(r['task_success']==(len(r['missing'])==0),'Stored grade contradicts missing-question list')
  for u in r['selected_evidence']:
   require(u['paper_id'] in catalog,'Unknown cited paper')
   require(type(u['text']) is str and bool(u['text']),'Empty selected quotation')
   require(hashlib.sha256(u['text'].encode()).hexdigest()==u['sentence_sha256'],'Quotation hash mismatch')
   base=catalog[u['paper_id']]['source_url'];require(u['source_url'].startswith(base+'#') and len(u['source_url'])>len(base)+1,'Citation version/source mismatch')
   quote_count+=1
  reviews[key]=r
 require(len(reviews)==24,'The historical study must retain all 24 reviews')
 summaries={s['arm']:s for s in o['summary']};require(set(summaries)==set(ARM_NAMES) and len(o['summary'])==3,'Missing/duplicate arm')
 fields=['tasks','completed','covered_requirements','requirements','model_calls','tool_calls','prompt_tokens','completion_tokens','reasoning_tokens','total_tokens']
 for arm,s in summaries.items():
  require(all(type(s[k]) is int and s[k]>=0 for k in fields),'Invalid cost integer')
  rr=[reviews[(arm,tid)] for tid in tasks]
  require(s['tasks']==8 and s['requirements']==32,'Registered denominator changed')
  require(sum(r['task_success'] for r in rr)==s['completed'],'Completion aggregate mismatch')
  require(32-sum(len(r['missing']) for r in rr)==s['covered_requirements'],'Question aggregate mismatch')
  require(s['total_tokens']==s['prompt_tokens']+s['completion_tokens'] and s['reasoning_tokens']<=s['completion_tokens'],'Token accounting mismatch')
 require(sum(s['model_calls'] for s in summaries.values())==o['actual_generations']==38,'Generation count mismatch')
 require(sum(s['tool_calls'] for s in summaries.values())==o['actual_tools']==102,'Tool count mismatch')
 pairs={p['task_id']:p for p in o['pairs']};require(set(pairs)==set(tasks) and len(o['pairs'])==8,'Pair set mismatch')
 for tid,p in pairs.items():
  for arm,field in [('qwen35_on_compiled','compiled_success'),('qwen35_on_two','two_success'),('qwen35_off_compiled','off_success')]:require(p[field]==reviews[(arm,tid)]['task_success'],'Pair grade mismatch')
 for arm,kind in [('qwen35_on_compiled','compiled'),('qwen35_on_two','two')]:
  require(sum(p[kind+'_calls'] for p in pairs.values())==summaries[arm]['model_calls'],'Pair call total mismatch')
  require(sum(p[kind+'_tokens'] for p in pairs.values())==summaries[arm]['total_tokens'],'Pair token total mismatch')
 a=summaries['qwen35_on_compiled'];c=summaries['qwen35_on_two'];call_reduction=1-a['model_calls']/c['model_calls'];token_reduction=1-a['total_tokens']/c['total_tokens'];comparison=o['comparisons']['same_model_on_two']
 require(math.isclose(call_reduction,comparison['actual_call_reduction_fraction']) and math.isclose(token_reduction,comparison['total_token_reduction_fraction']),'Reported reduction mismatch')
 t=b['training'];sft=t['r003-sft-provenance.json'];config=t['r003-sft-training-manifest.json'];expected=b['results']['sft']
 require(sft['actual_optimizer_steps_from_step_logs']==expected['optimizer_steps'] and sft['sample_visits']==expected['dataset_rows'],'SFT exposure mismatch')
 require(sft['adapter_sha256']==expected['adapter_sha256'] and sft['training_data_sha256']==config['data_sha256'],'SFT weight/data reference mismatch')
 dpo=t['r010-training-audit.json'];r10=t['r010-outcome.json']
 for seed in [43,44]:
  for kind in ['ct','dpo']:
   record=dpo[kind+str(seed)];require(record['seed']==seed and record['actual_steps']==51 and record['dataset_rows']==448 and record['changed_tensors']==392,'DPO/control recipe mismatch')
  require(dpo['dpo'+str(seed)]['reference_equal_sft_and_frozen_tensors']==392,'DPO reference check missing')
 require([r10['strict_totals']['fresh']['dpo'+str(s)] for s in [43,44]]==b['results']['matched_DPO']['DPO_strict'],'DPO result mismatch')
 require([r10['strict_totals']['fresh']['ct'+str(s)] for s in [43,44]]==b['results']['matched_DPO']['chosen_only_SFT_strict'],'DPO control result mismatch')
 grpo=t['r030-grpo-audit-excerpt.json']
 for arm,expected in b['results']['matched_GRPO'].items():
  require(all(grpo['training'][arm][k]==v for k,v in expected.items()),'GRPO record mismatch')
 require(grpo['registered_rule_met'] is False and grpo['source_sft_remains_default'] is True,'GRPO conclusion changed')
 return {'passed':True,'scope':'published record consistency, quote hashes and version URLs; no live original-source/weight/semantic revalidation','tasks':8,'arms':3,'reviews':24,'selected_quote_occurrences':quote_count,'recorded_generations':38,'recorded_tools':102,'same_mode_call_reduction':call_reduction,'same_mode_token_reduction':token_reduction,'same_mode_completion':[c['completed'],a['completed']],'training_records_crosschecked':6,'new_model_calls':0,'new_tool_calls':0,'new_optimizer_steps':0}

def render_browser(b,audit):
 q=html.escape;o=b['outcome'];summaries={s['arm']:s for s in o['summary']}
 rows=''.join('<tr><th>'+q(label)+'</th><td>'+str(summaries[arm]['completed'])+'/8</td><td>'+str(summaries[arm]['model_calls'])+'</td><td>'+format(summaries[arm]['total_tokens'],',')+'</td></tr>' for arm,label in ARM_NAMES.items())
 payload=json.dumps({'tasks':b['tasks'],'reviews':b['reviews'],'pairs':o['pairs'],'labels':ARM_NAMES},ensure_ascii=False).replace('<',chr(92)+'u003c')
 opts=''.join('<option value="'+q(t['id'])+'">'+q(t['id'])+'</option>' for t in b['tasks'])
 arms=''.join('<option value="'+q(arm)+'">'+q(label)+'</option>' for arm,label in ARM_NAMES.items())
 return '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>论文 Agent · 已记录实验证据</title><style>*{box-sizing:border-box}body{margin:0;background:#eef3f1;color:#18323d;font:17px/1.7 system-ui,sans-serif;overflow-wrap:anywhere}main{max-width:1100px;margin:auto;padding:30px 24px}header,section{background:white;border:1px solid #ccdcd9;border-radius:12px;padding:26px;margin-bottom:20px}h1{font-size:32px;line-height:1.35}h2{font-size:23px}a{color:#196e66}.note{background:#fff6e9;padding:15px;border-left:4px solid #b3824a}.muted{color:#516b77}.row{display:flex;gap:18px;flex-wrap:wrap}select{font:inherit;padding:8px;max-width:100%;border:1px solid #bccfce;border-radius:8px}table{width:100%;border-collapse:collapse}td,th{padding:12px 8px;border-bottom:1px solid #d5e1de;text-align:right}th:first-child{text-align:left}.scroll{overflow-x:auto;max-width:100%}.quote{border-left:3px solid #48877a;padding:13px 18px;background:#f4f8f6;margin:15px 0;white-space:pre-wrap}.source{font-size:14px}.badge{padding:7px 14px;background:#edf5f1;border-radius:7px}.fail{background:#fff0e1}.hash{font:12px/1.5 monospace;color:#5a7079}code{font-size:14px}@media(max-width:600px){main{padding:15px 12px}header,section{padding:19px}h1{font-size:27px}body{font-size:16px}td,th{font-size:14px}}</style></head><body><main><header><p class="muted">R056 历史结果浏览 · 本次模型调用 0 · 本次工具调用 0</p><h1>同一用户任务，查看答案证据与真实成本</h1><p>主问题：少一次模型规划，完整任务完成率能否保持？同 Qwen3.5、同开启思考模式，8 题均为 5/8；模型调用 16→11，全部 token 94,116→82,580。</p><p class="note">此页面浏览已记录的研究结果。完成标签来自原来的助手语义审阅，未在这里重新判题。样本复用六篇已知论文、答案为选取原句；绝对质量门槛未通过，不能据此承诺开放综述可靠性。</p><div class="scroll"><table id="summary"><thead><tr><th>路径</th><th>完整任务</th><th>记录的调用</th><th>全部 token</th></tr></thead><tbody>__ROWS__</tbody></table></div></header><section id="viewer"><h2>问题、缺项与选中的原句</h2><div class="row"><label>任务 <select id="task-select">__TASKS__</select></label><label>路径 <select id="arm-select">__ARMS__</select></label></div><p id="grade" class="badge"></p><p id="question"></p><p id="cost" class="muted"></p><p id="missing"></p><p id="review-note" class="muted"></p><div id="quotes"></div></section><section><h2>CPU 检查能证明什么</h2><p>公开的 24 条审阅记录、106 次选句记录、版本出处、训练记录引用和成本汇总相互一致；文本 SHA 匹配。重复选句按记录出现次数计数，不代表 106 个独立事实。</p><p>完整正文、输入/思考 token IDs、原始模型轨迹与权重位于实验服务器。此检查不重新下载原文，不比较权重，不替代完整原始审计，也不证明审阅标签正确。</p><p>真实训练是 Qwen2.5-1.5B 的 SFT/DPO/GRPO；4B 是原生能力与编排实验。DPO 两种子 45/66，继续 SFT 51/66、52/66；GRPO 未满足预注册收益与保持门槛。</p></section></main><script type="application/json" id="records">__DATA__</script><script>
const data=JSON.parse(document.getElementById('records').textContent);function show(){const tid=document.getElementById('task-select').value,arm=document.getElementById('arm-select').value,r=data.reviews.find(x=>x.task_id===tid&&x.arm===arm),t=data.tasks.find(x=>x.id===tid),p=data.pairs.find(x=>x.task_id===tid);const grade=document.getElementById('grade');grade.textContent=tid+' · '+data.labels[arm]+' · '+(r.task_success?'完整完成':'未完整完成');grade.className='badge'+(r.task_success?'':' fail');document.getElementById('question').textContent=t.task;document.getElementById('missing').textContent=r.missing.length?'未满足的问项：'+r.missing.join('；'):'原审阅记录中全部问项满足。';document.getElementById('review-note').textContent=r.note;document.getElementById('cost').textContent=arm==='qwen35_off_compiled'?'关闭思考路径公开逐题表未包含 token 明细，请使用顶部全批汇总。':'记录的实际模型调用 '+p[arm==='qwen35_on_two'?'two_calls':'compiled_calls']+'；全部 token '+p[arm==='qwen35_on_two'?'two_tokens':'compiled_tokens'].toLocaleString('en-US');const box=document.getElementById('quotes');box.replaceChildren();if(!r.selected_evidence.length){const text=document.createElement('p');text.textContent='此轨迹没有被接受的终答选句。未闭合或协议超限的生成成本已计入整批。';box.append(text);}for(const u of r.selected_evidence){const item=document.createElement('article'),quote=document.createElement('p'),source=document.createElement('a'),hash=document.createElement('p');quote.className='quote';quote.textContent=u.text;source.className='source';source.href=u.source_url;source.textContent=u.paper_id+' · 原始版本段落';source.target='_blank';source.rel='noopener noreferrer';hash.className='hash';hash.textContent='sentence SHA256 '+u.sentence_sha256;item.append(quote,source,hash);box.append(item);}}document.getElementById('task-select').addEventListener('change',show);document.getElementById('arm-select').addEventListener('change',show);show();
</script></body></html>'''.replace('__ROWS__',rows).replace('__TASKS__',opts).replace('__ARMS__',arms).replace('__DATA__',payload)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--html',type=Path,help='Write a standalone recorded-evidence browser; refuses existing paths');p.add_argument('--receipt',type=Path);a=p.parse_args()
 targets=[f for f in [a.html,a.receipt] if f is not None];require(len({f.resolve() for f in targets})==len(targets),'Output paths must differ')
 for f in targets:require(not f.exists(),'Refusing to overwrite '+str(f))
 b,hashes=load_bundle();audit=audit_bundle(b);audit['input_SHA256']=hashes
 if a.html:
  with a.html.open('x',encoding='utf-8') as f:f.write(render_browser(b,audit))
 if a.receipt:
  with a.receipt.open('x',encoding='utf-8') as f:f.write(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:v for k,v in audit.items() if k!='input_SHA256'},ensure_ascii=False))

if __name__=='__main__':main()
