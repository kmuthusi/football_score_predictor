import os
import subprocess
import sys
import unittest


class TestRLEvalSmoke(unittest.TestCase):
    def setUp(self):
        self.policy_path = os.path.join("models", "rl_policy_test.joblib")
        if not os.path.exists(self.policy_path):
            self.skipTest("Policy artifact not present (run RL train smoke test first)")

    def test_rl_eval_runs(self):
        subprocess.run([
            sys.executable,
            "rl_eval.py",
            "--policy",
            self.policy_path,
            "--artifact",
            "models/score_models.joblib",
            "--matches",
            "data/spi_matches.csv",
            "--stadiums",
            "data/stadium_coordinates_completed_full.csv",
        ], check=True)


if __name__ == "__main__":
    unittest.main()
