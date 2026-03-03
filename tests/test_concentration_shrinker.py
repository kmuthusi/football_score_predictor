import unittest
import numpy as np

from train import shrink_rho_for_concentration, top_score_mode_share


def make_edge_lambdas(n: int):
    # lambdas around 1 produce fairly concentrated 1-1 mode
    return np.ones(n) * 1.0, np.ones(n) * 1.0


class TestConcentrationShrinker(unittest.TestCase):
    def test_shrinker_reduces_share(self):
        lam_h, lam_a = make_edge_lambdas(200)
        max_goals = 3
        rho_initial = 0.3

        share_before = top_score_mode_share(lam_h, lam_a, max_goals, rho_initial, calibration_cfg=None)
        self.assertGreater(share_before, 0.5, "share should start above 50% with naive lambdas")

        rho_new, share_after = shrink_rho_for_concentration(
            lam_h, lam_a, rho_initial, max_goals, max_top_share=0.4, calibration_cfg=None
        )
        # shrinking rho should never increase the mode share, even if it cannot
        # bring it below the target (depending on the lambda distribution)
        self.assertLessEqual(share_after, share_before, "shrinker should not increase mode share")
        self.assertLessEqual(rho_new, rho_initial, "adjusted rho should not exceed original")


if __name__ == "__main__":
    unittest.main()
