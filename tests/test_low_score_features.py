import os
import json
import unittest

import pandas as pd

from features import FeatureConfig, build_training_frame, model_numeric_columns, load_matches_csv, load_stadiums_csv
from scripts.low_score_analysis import main as low_score_main


class TestLowScoreFeatures(unittest.TestCase):
    def setUp(self):
        # use sample data from repo; switch to completed stadium list with extras
        self.matches = load_matches_csv("data/spi_matches.csv")
        self.stadiums = load_stadiums_csv("data/stadium_coordinates_completed_full.csv")
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
        # adjusted goal-diff EWMA should also exist when enabled
        if self.cfg.use_adjusted_features and self.cfg.use_ewm_features:
            self.assertIn("home_adj_gd_ewm", self.frame.columns)
            self.assertIn("away_adj_gd_loc_ewm", self.frame.columns)
        # extra stadium features (city/altitude etc.) from completed file should propagate
        self.assertIn("home_city", self.frame.columns)
        self.assertIn("away_city", self.frame.columns)

    def test_numeric_columns_include_low_score(self):
        cols = model_numeric_columns(self.cfg)
        self.assertIn("home_low_score_mean_5", cols)
        self.assertIn("away_low_score_ewm", cols)
        # new goal-difference features should also be present
        self.assertIn("home_gd_mean_5", cols)
        self.assertIn("away_gd_ewm", cols)
        # and adjusted goal-diff
        if self.cfg.use_adjusted_features:
            self.assertIn("home_adj_gd_mean_5", cols)
            if self.cfg.use_ewm_features:
                # both sides should be present when EWMA used
                self.assertIn("home_adj_gd_ewm", cols)
                self.assertIn("away_adj_gd_loc_ewm", cols)

    def test_single_match_features_include_low_score(self):
        # build a dummy row via build_single_match_features and ensure low_score
        # metrics are present to avoid preprocessing validation errors in the app
        from features import build_team_match_long, build_single_match_features
        long_df = build_team_match_long(self.frame)
        asof = pd.Timestamp(self.matches["date"].max())
        row = build_single_match_features(
            div="1",
            home_team=self.matches.loc[0, "home_team"],
            away_team=self.matches.loc[0, "away_team"],
            asof_date=asof,
            home_odds=float(self.matches.loc[0, "home_odds"]),
            draw_odds=float(self.matches.loc[0, "draw_odds"]),
            away_odds=float(self.matches.loc[0, "away_odds"]),
            long_df=long_df,
            stadiums=self.stadiums,
            config=self.cfg,
        )
        for side in ("home", "away"):
            for w in self.cfg.windows:
                self.assertIn(f"{side}_low_score_mean_{w}", row.columns)
                self.assertIn(f"{side}_low_score_loc_mean_{w}", row.columns)
                # goal-diff should also appear
                self.assertIn(f"{side}_gd_mean_{w}", row.columns)
                self.assertIn(f"{side}_gd_loc_mean_{w}", row.columns)
        if self.cfg.use_ewm_features:
            for side in ("home", "away"):
                self.assertIn(f"{side}_low_score_ewm", row.columns)
                self.assertIn(f"{side}_low_score_loc_ewm", row.columns)
                self.assertIn(f"{side}_gd_ewm", row.columns)
                self.assertIn(f"{side}_gd_loc_ewm", row.columns)
            if self.cfg.use_adjusted_features:
                # adjusted goal-diff EWMA should also be preserved
                self.assertIn("home_adj_gd_ewm", row.columns)
                self.assertIn("away_adj_gd_loc_ewm", row.columns)

    def test_single_match_features_from_raw_matches(self):
        # simulate the app's long_df (no rolling features) and confirm no KeyError
        from features import build_team_match_long, build_single_match_features
        long_df = build_team_match_long(self.matches)
        asof = pd.Timestamp(self.matches["date"].max())
        # use first row's teams/odds to build row
        row = build_single_match_features(
            div=str(self.matches.loc[0, "div"]),
            home_team=self.matches.loc[0, "home_team"],
            away_team=self.matches.loc[0, "away_team"],
            asof_date=asof,
            home_odds=float(self.matches.loc[0, "home_odds"]),
            draw_odds=float(self.matches.loc[0, "draw_odds"]),
            away_odds=float(self.matches.loc[0, "away_odds"]),
            long_df=long_df,
            stadiums=self.stadiums,
            config=self.cfg,
        )
        # sanity: we still expect low_score columns even from raw long_df
        for col in ["home_low_score_mean_5", "away_low_score_ewm"]:
            self.assertIn(col, row.columns)
        # extra stadium fields should also be present after building single-row
        self.assertIn("home_city", row.columns)
        self.assertIn("away_city", row.columns)

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
