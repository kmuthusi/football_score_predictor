import os
import unittest
import joblib
import numpy as np

from rl_env import FootballBettingEnv, EnvConfig


class TestRLEnv(unittest.TestCase):
    def setUp(self):
        self.artifact_path = os.path.join("models", "score_models.joblib")
        if not os.path.exists(self.artifact_path):
            self.skipTest("models/score_models.joblib not present; skipping RL env integration tests")

    def test_reset_and_step(self):
        env = FootballBettingEnv(self.artifact_path, "data/spi_matches.csv", None, split="test")
        obs = env.reset()
        self.assertIsInstance(obs, np.ndarray)
        self.assertGreaterEqual(obs.size, 10)

        obs2, reward, done, info = env.step(0)
        self.assertIsInstance(obs2, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)
        self.assertIn("bankroll_after", info)


if __name__ == "__main__":
    unittest.main()
