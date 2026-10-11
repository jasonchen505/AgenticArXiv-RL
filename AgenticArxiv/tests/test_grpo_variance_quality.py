"""Zero relative advantage is not sufficient evidence of model failure."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('STORE_BACKEND','memory')
from rl.train_grpo import RewardVarianceGuard


class VarianceQualityTest(unittest.TestCase):
    def run_window(self,rewards,start=1):
        guard=RewardVarianceGuard(patience=5,grace_steps=20)
        control=SimpleNamespace(should_training_stop=False)
        for step,reward in enumerate(rewards,start):
            guard.on_log(None,SimpleNamespace(global_step=step),control,logs={'frac_reward_zero_std':1.,'reward_std':0.,'reward':reward,'rollout/finished':1.,'rollout/parse_error_rate':0.,'rollout/tool_error_rate':0.})
        return guard,control

    def test_early_full_score_window_stops_normally(self):
        guard,control=self.run_window([1.]*5,start=5)
        self.assertTrue(control.should_training_stop)
        self.assertTrue(guard.converged)
        self.assertFalse(guard.tripped)
        self.assertEqual(guard.stop_step,9)

    def test_early_floor_window_remains_failure(self):
        guard,control=self.run_window([-.75]*5)
        self.assertTrue(control.should_training_stop)
        self.assertTrue(guard.tripped)
        self.assertFalse(guard.converged)

    def test_late_floor_is_not_mislabeled_saturation(self):
        guard,_=self.run_window([-.75]*5,start=25)
        self.assertTrue(guard.tripped)
        self.assertFalse(guard.converged)

    def test_last_full_score_cannot_hide_bad_earlier_groups(self):
        guard,_=self.run_window([-.75,1.,1.,1.,1.],start=25)
        self.assertTrue(guard.tripped)
        self.assertFalse(guard.converged)

    def test_informative_group_resets_the_window(self):
        guard,_=self.run_window([1.]*4)
        control=SimpleNamespace(should_training_stop=False)
        guard.on_log(None,SimpleNamespace(global_step=5),control,logs={'frac_reward_zero_std':0.,'reward_std':.2,'reward':.5})
        guard.on_log(None,SimpleNamespace(global_step=6),control,logs={'frac_reward_zero_std':1.,'reward_std':0.,'reward':1.})
        self.assertEqual(guard.streak,1)
        self.assertFalse(control.should_training_stop)


if __name__=='__main__':unittest.main()
