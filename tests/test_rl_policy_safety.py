import unittest
import numpy as np

from rl_policy_utils import policy_is_safe


class TestRLPolicySafety(unittest.TestCase):
    def test_missing_safety_metadata_is_flagged(self):
        # policy trained with obs-norm but missing safety fields / obs_norm
        policy = {
            "W": np.zeros((4, 13)).tolist(),
            "b": np.zeros(4).tolist(),
            "train_cfg": {"use_obs_norm": True, "bet_penalty": None, "ev_threshold": None},
        }
        ok, reason = policy_is_safe(policy)
        self.assertFalse(ok)
        self.assertIn("bet_penalty", reason)
        self.assertIn("ev_threshold", reason)
        self.assertIn("obs_norm", reason)

    def test_policy_with_required_fields_passes(self):
        policy = {
            "W": np.zeros((4, 13)).tolist(),
            "b": np.zeros(4).tolist(),
            "train_cfg": {"use_obs_norm": True, "bet_penalty": 0.001, "ev_threshold": 0.01},
            "obs_norm": {"mean": [0.0] * 13, "std": [1.0] * 13},
        }
        ok, reason = policy_is_safe(policy)
        self.assertTrue(ok)
        self.assertEqual(reason, "")


if __name__ == "__main__":
    unittest.main()
