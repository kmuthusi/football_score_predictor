import os
import subprocess
import sys
import unittest
import joblib


class TestRLTrainSmoke(unittest.TestCase):
    def setUp(self):
        self.artifact_path = os.path.join("models", "score_models.joblib")
        if not os.path.exists(self.artifact_path):
            self.skipTest("models/score_models.joblib not present; skipping RL training smoke test")
        self.out_policy = os.path.join("models", "rl_policy_test.joblib")
        if os.path.exists(self.out_policy):
            os.remove(self.out_policy)

    def test_rl_train_creates_policy(self):
        subprocess.run([
            sys.executable,
            "rl_train.py",
            "--artifact",
            self.artifact_path,
            "--matches",
            "data/spi_matches.csv",
            "--stadiums",
            "data/stadium_coordinates_completed_full.csv",
            "--epochs",
            "1",
            "--save",
            self.out_policy,
            "--no-test-eval",
        ], check=True)

        self.assertTrue(os.path.exists(self.out_policy))
        policy = joblib.load(self.out_policy)
        self.assertIn("W", policy)
        self.assertIn("b", policy)
        # obs_norm stats should be saved when training used observation normalization
        if policy.get("train_cfg", {}).get("use_obs_norm", False):
            self.assertIn("obs_norm", policy)
            self.assertIn("mean", policy["obs_norm"])
            self.assertIn("std", policy["obs_norm"])


if __name__ == "__main__":
    unittest.main()
