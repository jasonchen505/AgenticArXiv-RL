import copy,unittest
from portfolio.review_results import load_bundle,audit_bundle

class PublishedRecordTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.bundle,_=load_bundle()
 def test_full_published_records(self):
  r=audit_bundle(self.bundle);self.assertEqual((r['reviews'],r['selected_quote_occurrences'],r['new_model_calls']),(24,106,0))
 def test_reject_modified_quotation(self):
  b=copy.deepcopy(self.bundle);r=next(x for x in b['reviews'] if x['selected_evidence']);r['selected_evidence'][0]['text']+=' changed'
  with self.assertRaisesRegex(ValueError,'Quotation hash'):audit_bundle(b)
 def test_reject_wrong_paper_version(self):
  b=copy.deepcopy(self.bundle);r=next(x for x in b['reviews'] if x['selected_evidence']);r['selected_evidence'][0]['source_url']='https://arxiv.org/html/2310.12823v2#fake'
  with self.assertRaisesRegex(ValueError,'Citation version'):audit_bundle(b)
 def test_reject_missing_failed_case(self):
  b=copy.deepcopy(self.bundle);b['reviews']=[r for r in b['reviews'] if r['task_success']]
  with self.assertRaisesRegex(ValueError,'retain all 24'):audit_bundle(b)
 def test_reject_pair_cost_tampering(self):
  b=copy.deepcopy(self.bundle);b['outcome']['pairs'][0]['compiled_tokens']+=1;b['results']['qwen35_body_study']=copy.deepcopy(b['outcome'])
  with self.assertRaisesRegex(ValueError,'Pair token'):audit_bundle(b)
 def test_reject_training_claim_drift(self):
  b=copy.deepcopy(self.bundle);b['training']['r003-sft-provenance.json']['adapter_sha256']='0'*64
  with self.assertRaisesRegex(ValueError,'SFT weight'):audit_bundle(b)

if __name__=='__main__':unittest.main()
