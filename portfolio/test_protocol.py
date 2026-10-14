import unittest
from portfolio.paper_agent.identity import resolve
from portfolio.paper_agent.protocol import parse_selection,parse_plan,status_from_events
class ContractTests(unittest.TestCase):
 def setUp(self):self.cat=[{'alias':'Alpha','paper_id':'2301.00001v1','title':'Alpha'},{'alias':'Beta','paper_id':'2301.00002v1','title':'Beta'}]
 def test_public_identity_and_real_fallback(self):
  self.assertEqual(resolve('请比较Alpha与Beta',self.cat)[0]['paper_ids'],['2301.00001v1','2301.00002v1'])
  self.assertIsNone(resolve('请比较Alpha与Beta，不要忽略限制',self.cat)[0])
  self.assertIsNone(resolve('请查2301.00001v1与2301.00002v1',self.cat)[0])
 def test_protocol_rejects_not_repairs(self):
  for text in ['{"sentence_ids":[0,0]}','{"sentence_ids":[true]}','{"sentence_ids":[9]}','{"sentence_ids":['+','.join(str(i) for i in range(9))+']}','```json\n{"sentence_ids":[0]}\n```']:
   with self.assertRaises(ValueError):parse_selection(text,[{}]*9)
  with self.assertRaises(ValueError):parse_plan('{"paper_ids":["wrong"]}',self.cat)
 def test_execution_status_requires_matching_success(self):
  def ev(n,p,s):return {'name':n,'args':{'paper_id':p},'success':s}
  self.assertIn('不能报告读取成功',status_from_events([]))
  self.assertIn('尚未完成',status_from_events([ev('read_paper','A',False),ev('read_cached_paper','B',True)]))
  self.assertIn('恢复成功',status_from_events([ev('read_paper','A',False),ev('read_cached_paper','A',True)]))
if __name__=='__main__':unittest.main()
