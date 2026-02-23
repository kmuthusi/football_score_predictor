import os
import json
import unittest

import pandas as pd

from features import FeatureConfig, build_training_frame, model_numeric_columns, load_matches_csv, load_stadiums_csv
from scripts.low_score_analysis import main as low_score_main


class TestLowScoreFeatures(unittest.TestCase):
    def setUp(self):
        # use sample data from repo
        self.matches = load_matches_csv("data/spi_matches.csv")
        self.stadiums = load_stadiums_csv("data/stadium_coordinates.csv")
        self.cfg = FeatureConfig()
        self.frame = build_training_frame(self.matches, self.stadiums, self.cfg)

    def test_low_score_columns_exist(self):
        # verify rolling and ewm features for low scores are present
        for side in ("home", "away"):
            for w in self.cfg.windows:
                self.assertIn(f"{side}_low_score_mean_{w}", self.frame.columns)
                self.assertIn(f"{side}_low_score_loc_mean_{w}", self.frame.columns)
            if self.cfg.use_ewm_features:
                self.assertIn(f"{side}_low_score_ewm", self.frame.columns)
                self.assertIn(f"{side}_low_score_loc_ewm", self.frame.columns)

    def test_numeric_columns_include_low_score(self):
        cols = model_numeric_columns(self.cfg)
        self.assertIn("home_low_score_mean_5", cols)
        self.assertIn("away_low_score_ewm", cols)

    def test_low_score_analysis_outputs_json(self):
        # run script, capture output JSON file
        out_path = "reports/low_score_bias.json"
        if os.path.exists(out_path):
            os.remove(out_path)
        # call main as module; it should create the file
        try:
            low_score_main()
        except SystemExit:
            pass
        self.assertTrue(os.path.exists(out_path))
        with open(out_path) as f:
            data = json.load(f)
        self.assertIn("obs_low1", data)
        self.assertIn("bias_low1", data)


if __name__ == "__main__":
    unittest.main()
