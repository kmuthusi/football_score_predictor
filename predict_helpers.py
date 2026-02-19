"""Helpers for preparing and validating model inputs used at prediction time.

Provides a small, well-tested wrapper that builds leakage-safe features for a
single match and validates the resulting DataFrame contains exactly the
`artifact["cat_cols"] + artifact["num_cols"]` expected by the saved pipeline.
"""
from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from features import FeatureConfig, build_single_match_features, build_team_match_long


def prepare_and_validate_row(
    artifact: Dict[str, object],
    matches: pd.DataFrame,
    stadiums: Optional[pd.DataFrame],
    div: str,
    home_team: str,
    away_team: str,
    asof_date: pd.Timestamp | str,
    home_odds: float | None,
    draw_odds: float | None,
    away_odds: float | None,
) -> pd.DataFrame:
    """Build the single-row feature frame used by the model and validate columns.

    Returns a DataFrame with columns ordered as `artifact['cat_cols'] + artifact['num_cols']`.

    Raises:
      ValueError: if the produced feature frame is missing any required column.
    """
    cfg = FeatureConfig(**artifact.get("config", {}))

    # Build long-form team history (leakage-safe) and construct the single-row
    long_df = build_team_match_long(matches.copy())

    asof_ts = pd.Timestamp(asof_date)
    feat_row = build_single_match_features(
        div=div,
        home_team=home_team,
        away_team=away_team,
        asof_date=asof_ts,
        home_odds=home_odds,
        draw_odds=draw_odds,
        away_odds=away_odds,
        long_df=long_df,
        stadiums=stadiums if stadiums is not None else pd.DataFrame(),
        config=cfg,
    )

    required_cols = list(artifact.get("cat_cols", [])) + list(artifact.get("num_cols", []))
    missing = [c for c in required_cols if c not in feat_row.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    X = feat_row[required_cols].copy()

    # Ensure categorical columns are strings (model pipeline expects object dtype)
    for c in artifact.get("cat_cols", []):
        if c in X.columns:
            X[c] = X[c].astype(str)

    return X
