"""Verify selective cap, legacy parity, and preservation of reward evidence."""
import copy
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'AgenticArxiv'))
from rl.grpo_reward import make_grpo_reward_fn
from rl.reward import RewardCalculator
from benchmark.metrics import extract_metrics,is_strict_success

class Capture:
    def record_batch(self,**kwargs):self.batch=kwargs

class BlockedReasonObjectiveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[2]
        cls.tasks={x['id']:x for x in json.loads((root/'artifacts/reproduction/round-002-2026-10-02/task-oracles.json').read_text())}

    def score(self,tid,result,objective='legacy'):
        capture=Capture()
        fn=make_grpo_reward_fn(self.tasks,reward_calc=RewardCalculator(curriculum_steps=0),auditor=capture,reward_objective=objective)
        reward=fn(completions=['unused'],task_id=[tid],trajectory_results=[result])[0]
        return reward,capture.batch

    @staticmethod
    def finish(text):return {'history':[{'thought':text,'action':'FINISH','observation':'任务完成'}]}

    def test_wrong_reason_loses_positive_credit(self):
        result=self.finish('指定的 arXiv 论文 ID 不在快照记录中，无法下载该论文')
        original=copy.deepcopy(result)
        old,_=self.score('infeasible_zero_index',result)
        new,batch=self.score('infeasible_zero_index',result,'blocked_reason_v1')
        self.assertEqual(old,.7);self.assertEqual(new,0)
        self.assertEqual(result,original)
        audit=batch['trajectories'][0]['training_reward_audit']
        self.assertTrue(audit['gate_triggered']);self.assertTrue(audit['reward_changed'])
        self.assertEqual(audit['legacy_reward'],old)
        self.assertEqual(batch['breakdowns'][0].total,new)

    def test_correct_blocked_reason_stays_full_reward(self):
        result=self.finish('该 ref 序号违反 1-based 索引范围，属于非法取值，不能执行')
        self.assertEqual(self.score('infeasible_zero_index',result)[0],1)
        new,batch=self.score('infeasible_zero_index',result,'blocked_reason_v1')
        self.assertEqual(new,1);self.assertFalse(batch['trajectories'][0]['training_reward_audit']['gate_triggered'])

    def test_negative_tool_failure_is_not_lifted(self):
        result={'history':[{'action':json.dumps({'name':'download_arxiv_pdf','args':{'ref':0}}),'thought':'调用下载','observation':'工具执行失败: ref 必须从 1 开始'}, {'action':'FINISH','thought':'所有步骤均已成功执行，任务已完成','observation':'任务完成'}]}
        old,_=self.score('infeasible_zero_index',result)
        new,_=self.score('infeasible_zero_index',result,'blocked_reason_v1')
        self.assertLess(old,0);self.assertEqual(new,old)

    def test_feasible_partial_failure_and_default_stay_identical(self):
        result=self.finish('无法找到指定论文，不能继续')
        tid='search_CR_7d_5'
        old,_=self.score(tid,result);new,batch=self.score(tid,result,'blocked_reason_v1')
        calc=RewardCalculator(curriculum_steps=0)
        direct,_=calc.compute_reward_breakdown(self.tasks[tid],result)
        self.assertEqual(old,direct.total);self.assertEqual(old,new)
        self.assertFalse(batch['trajectories'][0]['training_reward_audit']['gate_triggered'])

    def test_all_archived_train_rows_replay_legacy_exactly(self):
        root=Path(__file__).resolve().parents[2]
        for directory,arm in [('round-006-2026-10-03','hard7'),('round-007-2026-10-04','hard44')]:
            for line in (root/'artifacts/reproduction'/directory/(arm+'-rollout-traces.jsonl')).read_text().splitlines():
                row=json.loads(line)
                with self.subTest(task=row['task_id'],step=row['training_step'],generation=row['generation_index']):
                    self.assertEqual(self.score(row['task_id'],row['trajectory'])[0],row['reward'])

    def test_unknown_objective_rejected(self):
        with self.assertRaises(ValueError):make_grpo_reward_fn(self.tasks,reward_objective='typo')

if __name__=='__main__':unittest.main()
