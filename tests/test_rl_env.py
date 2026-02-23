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
        # new low1 feature adds an extra dimension
        self.assertEqual(obs.size, 14)

        obs2, reward, done, info = env.step(0)
        self.assertIsInstance(obs2, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)
        self.assertIn("bankroll_after", info)

    def test_run_episode_early_stop_on_min_stake(self):
        # Construct minimal arrays for run_episode and force a bet action so stake < min_stake
        from rl_train import run_episode, TrainConfig
        rng = np.random.default_rng(0)

        n = 5
        arrays = {
            "pH": np.full(n, 0.5),
            "pD": np.full(n, 0.25),
            "pA": np.full(n, 0.25),
            "impH": np.full(n, 0.4),
            "impD": np.full(n, 0.3),
            "impA": np.full(n, 0.3),
            "ho": np.full(n, 2.5),
            "do": np.full(n, 3.0),
            "ao": np.full(n, 3.5),
            "outcome": np.zeros(n, dtype=int),
        }

        cfg = TrainConfig(epochs=1, initial_bankroll=2.0, stake_frac=0.01, min_bankroll=1.0, min_stake=1.0, ev_threshold=0.0)
        obs_dim = 14
        W = np.zeros((4, obs_dim))
        b = np.array([-10.0, 10.0, -10.0, -10.0])

        obs_mat, probs_mat, actions, stats = run_episode(arrays, W=W, b=b, rng=rng, cfg=cfg, greedy=True)
        # Episode should end immediately because stake (bankroll*stake_frac=0.02) < min_stake (1.0)
        self.assertLessEqual(int(stats["n_steps"]), 1)

    def test_bet_penalty_reduces_sum_reward(self):
        from rl_train import run_episode, TrainConfig
        rng = np.random.default_rng(1)

        n = 4
        arrays = {
            "pH": np.array([0.5, 0.5, 0.5, 0.5]),
            "pD": np.array([0.25, 0.25, 0.25, 0.25]),
            "pA": np.array([0.25, 0.25, 0.25, 0.25]),
            "impH": np.array([0.4, 0.4, 0.4, 0.4]),
            "impD": np.array([0.3, 0.3, 0.3, 0.3]),
            "impA": np.array([0.3, 0.3, 0.3, 0.3]),
            "ho": np.array([2.5, 2.5, 2.5, 2.5]),
            "do": np.array([3.0, 3.0, 3.0, 3.0]),
            "ao": np.array([3.5, 3.5, 3.5, 3.5]),
            "outcome": np.array([0, 0, 0, 0], dtype=int),
        }

        cfg_no_pen = TrainConfig(epochs=1, initial_bankroll=1000.0, stake_frac=0.01, min_bankroll=1.0, min_stake=0.01, ev_threshold=0.0, bet_penalty=0.0)
        cfg_with_pen = TrainConfig(epochs=1, initial_bankroll=1000.0, stake_frac=0.01, min_bankroll=1.0, min_stake=0.01, ev_threshold=0.0, bet_penalty=0.05)

        # also test low_score_penalty effect
        cfg_low_pen = TrainConfig(epochs=1, initial_bankroll=1000.0, stake_frac=0.01, min_bankroll=1.0, min_stake=0.01, ev_threshold=0.0, low_score_penalty=0.1)

        obs_dim = 14
        W = np.zeros((4, obs_dim))
        b = np.array([-10.0, 10.0, -10.0, -10.0])

        _, _, _, stats_no = run_episode(arrays, W=W, b=b, rng=rng, cfg=cfg_no_pen, greedy=True)
        _, _, _, stats_pen = run_episode(arrays, W=W, b=b, rng=rng, cfg=cfg_with_pen, greedy=True)
        _, _, _, stats_low = run_episode(arrays, W=W, b=b, rng=rng, cfg=cfg_low_pen, greedy=True)

        # With a per-bet penalty the sum_reward should be lower (more negative)
        self.assertLess(stats_pen["sum_reward"], stats_no["sum_reward"])
        # low-score penalty should also reduce reward
        self.assertLess(stats_low["sum_reward"], stats_no["sum_reward"])


if __name__ == "__main__":
    unittest.main()
