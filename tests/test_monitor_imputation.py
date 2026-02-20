import os
import json
import unittest

from monitor_imputation import summarize_imputation


class TestMonitorImputation(unittest.TestCase):
    def setUp(self):
        self.artifact = os.path.join("models", "score_models.joblib")
        self.matches = os.path.join("data", "spi_matches.csv")
        if not os.path.exists(self.artifact) or not os.path.exists(self.matches):
            self.skipTest("Required data/artifact not available; skipping monitor_imputation tests")

    def test_summarize_imputation_outputs_expected_keys(self):
        rpt = summarize_imputation(self.artifact, self.matches, None, recent_days=30, out="imputation_report_test.json")
        self.assertIn("rows_with_any_imputation_prop", rpt)
        self.assertIn("per_column_missing_rate", rpt)
        self.assertIn("generated_at", rpt)
        # cleanup
        try:
            os.remove("imputation_report_test.json")
        except Exception:
            pass


if __name__ == '__main__':
    unittest.main()
