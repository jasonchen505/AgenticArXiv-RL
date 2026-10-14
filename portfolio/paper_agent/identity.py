import re,unicodedata
def norm(s):return unicodedata.normalize('NFKC',s).replace('∗','*').replace('–','-').replace('—','-').casefold().strip()
def resolve(task,catalog):
    text=norm(task);by={}
    for c in catalog:by.setdefault(norm(c['alias']),[]).append(c['paper_id'])
    hits=[]
    for alias,pids in by.items():
        for m in re.finditer(r'(?<![a-z0-9])'+re.escape(alias)+r'(?![a-z0-9])',text):hits.append((m.start(),alias,pids))
    mentioned=list(dict.fromkeys(a for _,a,_ in sorted(hits)))
    def reject(reason):return None,{'eligible':False,'reason':reason,'inputs_only':'original user request and public catalog'}
    if len(mentioned)!=2:return reject('requires exactly two explicit public aliases; otherwise model fallback')
    if any(len(by[a])!=1 for a in mentioned):return reject('ambiguous public alias')
    if any(w in text for w in ['不要','不需要','不讨论','忽略','排除','不引用','只是例子','仅为背景','只是背景']):return reject('negation or incidental-reference syntax needs model fallback')
    if not any(w in text for w in ['请','帮','核对','核查','比较','整理','我想','我要','给我']):return reject('no explicit request intent')
    # Marked unknown names cannot silently disappear behind the two known names.
    for name in re.findall(r'(?:关于|从)\s*([a-z][a-z0-9.*\- ]*?)(?=[，,:：中里])',text):
        if norm(name) not in by:return reject('marked unknown alias')
    ids=[by[a][0] for a in mentioned]
    for aid in re.findall(r'(?<![0-9a-z])\d{4}\.\d{4,5}(?:v\d+)?(?![0-9a-z])',text):
        if len([pid for pid in ids if pid==aid or 'v' not in aid and pid.split('v')[0]==aid])!=1:return reject('unknown or conflicting explicit arXiv version')
    if any(re.search(re.escape(a)+r'\s*\(?v\d+',text) for a in mentioned):return reject('alias-local version syntax requires model fallback')
    return {'paper_ids':ids},{'eligible':True,'aliases':mentioned,'public_paper_ids':ids,'inputs_only':'original user request and public catalog','no_private_gold_or_tool_result_used':True,'scope':'exactly two explicit unique aliases with request intent; conservative fallback for indirect, negated and version syntax'}
