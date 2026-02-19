from __future__ import annotations

import unittest

import numpy as np

from metrics import neg_log_likelihood, neg_log_likelihood_dixon_coles
from predict import apply_temperature_scaling, scoreline_probability_matrix


class TestScientificChecks(unittest.TestCase):
    def test_dc_nll_equals_independent_when_rho_zero(self) -> None:
        y_home = np.array([0, 1, 2, 3, 1, 0], dtype=int)
        y_away = np.array([0, 1, 1, 2, 0, 2], dtype=int)
        lam_home = np.array([0.8, 1.2, 1.9, 2.2, 1.1, 0.7], dtype=float)
        lam_away = np.array([0.7, 1.1, 1.4, 1.6, 0.9, 1.8], dtype=float)

        nll_ind = neg_log_likelihood(y_home, y_away, lam_home, lam_away)
        nll_dc = neg_log_likelihood_dixon_coles(y_home, y_away, lam_home, lam_away, rho=0.0)

        self.assertAlmostEqual(nll_ind, nll_dc, places=12)

    def test_scoreline_matrix_is_normalized_with_and_without_dc(self) -> None:
        mat_ind = scoreline_probability_matrix(1.42, 0.97, max_goals=6, include_tail_bucket=True, rho=None)
        mat_dc = scoreline_probability_matrix(1.42, 0.97, max_goals=6, include_tail_bucket=True, rho=-0.08)

        self.assertAlmostEqual(float(mat_ind.to_numpy().sum()), 1.0, places=12)
        self.assertAlmostEqual(float(mat_dc.to_numpy().sum()), 1.0, places=12)

    def test_temperature_scaling_preserves_rank_order(self) -> None:
        probs = np.array([0.60, 0.25, 0.10, 0.05], dtype=float)
        rank_base = np.argsort(-probs)

        for temperature in (0.7, 1.0, 1.4):
            scaled = apply_temperature_scaling(probs, temperature=temperature)
            rank_scaled = np.argsort(-scaled)
            self.assertTrue(np.array_equal(rank_base, rank_scaled))

    def test_temperature_sharpens_and_smooths_as_expected(self) -> None:
        probs = np.array([0.70, 0.20, 0.10], dtype=float)
        sharp = apply_temperature_scaling(probs, temperature=0.7)
        smooth = apply_temperature_scaling(probs, temperature=1.4)

        self.assertGreater(float(sharp[0]), float(probs[0]))
        self.assertLess(float(smooth[0]), float(probs[0]))


if __name__ == "__main__":
    unittest.main()
