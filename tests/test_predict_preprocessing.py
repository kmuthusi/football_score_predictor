import unittest

import pandas as pd

from features import load_matches_csv, load_stadiums_csv
from predict import load_artifact
from predict_helpers import prepare_and_validate_row


class TestPredictPreprocessing(unittest.TestCase):
    def test_prepare_row_matches_artifact_columns(self):
        artifact = load_artifact("models/score_models.joblib")
        matches = load_matches_csv("data/spi_matches.csv")
        stadiums = load_stadiums_csv("data/stadium_coordinates_completed_full.csv")

        # Pick a realistic row with odds present
        row = matches.dropna(subset=["home_odds", "draw_odds", "away_odds"]).iloc[0]
        div = row["div"]
        home = row["home_team"]
        away = row["away_team"]
        asof_date = pd.Timestamp(row["date"])
        home_odds = float(row["home_odds"])
        draw_odds = float(row["draw_odds"])
        away_odds = float(row["away_odds"])

        X = prepare_and_validate_row(
            artifact=artifact,
            matches=matches,
            stadiums=stadiums,
            div=div,
            home_team=home,
            away_team=away,
            asof_date=asof_date,
            home_odds=home_odds,
            draw_odds=draw_odds,
            away_odds=away_odds,
        )

        expected_cols = artifact["cat_cols"] + artifact["num_cols"]
        self.assertListEqual(list(X.columns), list(expected_cols))


if __name__ == "__main__":
    unittest.main()
