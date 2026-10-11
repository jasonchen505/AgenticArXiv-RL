"""A truncated model view must not destroy the environment's scoring evidence."""
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('STORE_BACKEND','memory')
from rl.grpo_reward import make_multiturn_rollout_func
from rl.reward import RewardCalculator


class Tokenizer:
    def apply_chat_template(self,prompt,**kwargs):
        return [ord(c) for c in 'Task']
    def __call__(self,text,**kwargs):
        return {'input_ids':[ord(c) for c in text]}
    def decode(self,ids,**kwargs):
        return ''.join(chr(i) for i in ids)


class Trainer:
    processing_class=Tokenizer()
    max_completion_length=5000
    def __init__(self):
        self.turn=0
        self.visible_prompts=[]
    def _generate_single_turn(self,prompts,*args):
        self.visible_prompts.extend(self.processing_class.decode(p) for p in prompts)
        text='Thought: analyze\nAction: '+json.dumps({'name':'analyze_figure','args':{'ref':1,'figure_ref':2,'question_type':'axes'}}) if self.turn==0 else 'Thought: done\nAction: FINISH'
        self.turn+=1
        return [[ord(c) for c in text] for _ in prompts],None,None


class FullObservationTest(unittest.TestCase):
    def run_figure(self,answer):
        payload={'paper_id':'test-paper','answer':answer,'extra':'x'*1400}
        class Environment:
            def reset(self,**kwargs):pass
            def analyze_figure(self,**kwargs):return payload
        trainer=Trainer()
        with patch('rl.grpo_reward.require_rollout_func_support'):
            output=make_multiturn_rollout_func(Environment,max_turns=2)([[{'role':'user','content':'Analyze'}]],trainer)
        task={'id':'figure','expected_tools':['analyze_figure'],'expected_tool_args':[{'ref':1,'figure_ref':2,'question_type':'axes'}]}
        trajectory=output['trajectory_results'][0]
        reward,_=RewardCalculator(curriculum_steps=0).compute_reward_breakdown(task,trajectory)
        return payload,trainer,output,trajectory,reward

    def test_long_valid_answer_keeps_full_scoring_evidence(self):
        payload,trainer,output,trajectory,reward=self.run_figure('a'*1400+'ANSWER_END')
        self.assertEqual(trajectory['history'][0]['observation'],str(payload))
        self.assertEqual(reward.total,1.0)
        self.assertNotIn('ANSWER_END',trainer.visible_prompts[1])
        self.assertIn(0,output['env_mask'][0])
        self.assertEqual(sum(output['env_mask'][0]),trajectory['token_usage']['policy_tokens'])

    def test_real_empty_answer_remains_a_failure(self):
        _,_,_,_,reward=self.run_figure('')
        self.assertLessEqual(reward.total,-0.75)

    def test_model_view_remains_character_bounded(self):
        payload,_,_,trajectory,_=self.run_figure('axis labels')
        self.assertEqual(trajectory['history'][0]['model_observation'],str(payload)[:1000])


if __name__=='__main__':unittest.main()
