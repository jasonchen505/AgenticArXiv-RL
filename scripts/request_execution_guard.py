"""Instruction-only prototype for the explicit Chinese workflow grammar.

No benchmark metadata, oracle, model weights or tool invocation is accepted.
The caller proposes actions and supplies actual successful tool observations.
Rejected proposals and failed observations never advance execution state.
Unsupported free-form requests must use the existing agent unchanged.
"""
from dataclasses import dataclass
import json,re

class UnsupportedRequest(ValueError):pass

@dataclass(frozen=True)
class Step:
    name:str
    args:dict

@dataclass(frozen=True)
class Decision:
    allowed:bool
    reason:str

PREFIX=re.compile(r'检索最近(\d+)天(?:人工智能\(cs\.(AI)\)|计算机视觉\(cs\.(CV)\))论文(\d+)篇，下载第(\d+)篇，然后按以下顺序处理同一篇论文：(.+)。完成这些步骤后结束。')
SUMMARY=re.compile(r'以(tldr|structured)风格用不超过(\d+)词总结它')
ANALYSIS=re.compile(r'描述它的第(\d+)张图')

def compile_request(text:str):
    if not isinstance(text,str):raise UnsupportedRequest('Only user instruction text is accepted')
    m=PREFIX.fullmatch(text.strip())
    if not m:raise UnsupportedRequest('Outside supported explicit workflow grammar')
    days,ai,cv,count,ref,body=m.groups();days,count,ref=int(days),int(count),int(ref)
    if min(days,count,ref)<1 or ref>count:raise UnsupportedRequest('Invalid requested search or paper slot')
    steps=[Step('get_recently_submitted_cs_papers',{'aspect':ai or cv,'days':days,'max_results':count}),Step('download_arxiv_pdf',{'ref':ref})]
    for clause in body.split('；'):
        if clause=='读取它默认返回的标题与摘要':s=Step('get_paper_content',{'ref':ref,'section':None})
        elif clause=='抽取它的全部图表':s=Step('extract_paper_figures',{'ref':ref})
        elif SUMMARY.fullmatch(clause):
            style,words=SUMMARY.fullmatch(clause).groups();words=int(words)
            if words<1:raise UnsupportedRequest('Invalid summary word budget')
            s=Step('summarize_paper',{'ref':ref,'style':style,'max_words':words})
        elif ANALYSIS.fullmatch(clause):
            figure=int(ANALYSIS.fullmatch(clause).group(1))
            if figure<1 or not any(x.name=='extract_paper_figures' for x in steps):raise UnsupportedRequest('Figure analysis needs prior extraction')
            s=Step('analyze_figure',{'ref':ref,'figure_no':figure,'question':'describe'})
        else:raise UnsupportedRequest('Unsupported requested operation')
        steps.append(s)
    return tuple(steps)

class RequestExecutionGuard:
    def __init__(self,user_instruction:str):
        self.steps=compile_request(user_instruction);self.position=0;self.selected_paper_id=None;self.downloaded_paper_id=None;self.finished=False
    def status(self):
        return {'completed_steps':self.position,'requested_steps':len(self.steps),'ready_to_finish':self.position==len(self.steps),'finished':self.finished}
    def _decode(self,action):
        if action=='FINISH':return None
        try:x=json.loads(action) if isinstance(action,str) else action
        except (TypeError,ValueError):raise ValueError('Malformed tool proposal')
        if not isinstance(x,dict) or set(x)!={'name','args'} or not isinstance(x['name'],str) or not isinstance(x['args'],dict):raise ValueError('Malformed tool proposal')
        return x
    def _normalized_args(self,x):
        a=dict(x['args']);expected=self.steps[self.position].args
        if 'paper_id' in a:
            if not self.downloaded_paper_id or a.pop('paper_id')!=self.downloaded_paper_id:raise ValueError('Wrong or unbound paper identity')
            if 'ref' in a and a['ref']!=expected.get('ref'):raise ValueError('Conflicting paper slot')
            a['ref']=expected.get('ref')
        # Missing null section is the tool's documented default; all user-specified values remain exact.
        if expected.get('section','not-null') is None:a.setdefault('section',None)
        return a
    def check(self,action):
        if self.finished:return Decision(False,'Request already finished')
        if action=='FINISH':return Decision(self.position==len(self.steps),'All requested steps succeeded' if self.position==len(self.steps) else 'Requested steps remain unfinished')
        if self.position==len(self.steps):return Decision(False,'All requested tool steps already succeeded')
        try:
            x=self._decode(action)
            if x['name']!=self.steps[self.position].name:return Decision(False,'Tool is not the next operation requested by the user')
            if self._normalized_args(x)!=self.steps[self.position].args:return Decision(False,'Arguments do not match the user instruction')
        except ValueError as e:return Decision(False,str(e))
        return Decision(True,'Proposal matches the next requested operation')
    def observe(self,action,result):
        """Commit only after an allowed action and actual successful tool response."""
        decision=self.check(action)
        if not decision.allowed:return decision
        if action=='FINISH':self.finished=True;return Decision(True,'Finished after verified requested steps')
        name=self.steps[self.position].name
        if isinstance(result,str) or (isinstance(result,dict) and (result.get('error') or result.get('status') in ['ERROR','FAILED'])):return Decision(False,'Tool did not supply a successful structured observation')
        if name=='get_recently_submitted_cs_papers':
            target=self.steps[1].args['ref']
            if not isinstance(result,list) or len(result)<target or not isinstance(result[target-1],dict) or not result[target-1].get('id'):return Decision(False,'Requested paper slot is unavailable')
            selected=result[target-1]['id']
        else:
            if not isinstance(result,dict) or not self.selected_paper_id or result.get('paper_id')!=self.selected_paper_id:return Decision(False,'Observation refers to a different paper')
            if name=='download_arxiv_pdf' and result.get('status')!='READY':return Decision(False,'Requested paper is not ready')
            if name=='extract_paper_figures' and ('count' not in result or not isinstance(result['count'],int)):return Decision(False,'No validated figure extraction result')
            if name=='get_paper_content' and not result.get('content'):return Decision(False,'Requested paper content is unavailable')
            if name=='summarize_paper' and not result.get('summary'):return Decision(False,'Requested summary is unavailable')
            if name=='analyze_figure' and not (result.get('answer') or result.get('analysis') or result.get('description')):return Decision(False,'Requested figure description is unavailable')
        # Every return above is transactional: no state mutation on denied/error responses.
        if name=='get_recently_submitted_cs_papers':self.selected_paper_id=selected
        if name=='download_arxiv_pdf':self.downloaded_paper_id=result['paper_id']
        self.position+=1;return Decision(True,'Successful requested step committed')
