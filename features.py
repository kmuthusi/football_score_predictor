from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    windows: Tuple[int, ...] = (5, 10)
    max_goals: int = 4
    use_travel_distance: bool = True
    ewm_span: int = 6
    use_ewm_features: bool = True
    use_adjusted_features: bool = True


_MATCH_REQUIRED_COLS = [
    "div", "date", "home_team", "away_team",
    "home_goals", "away_goals",
    "home_odds", "draw_odds", "away_odds",
]

_ODDS_COLS = ["home_odds", "draw_odds", "away_odds"]
_GOAL_COLS = ["home_goals", "away_goals"]


def preprocess_matches_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team", "div"])
    df["div"] = df["div"].astype(str)
    df["home_team"] = df["home_team"].astype(str)
    df["away_team"] = df["away_team"].astype(str)
    for c in _GOAL_COLS + _ODDS_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values(["date", "div", "home_team", "away_team"]).reset_index(drop=True)
    df["match_id"] = np.arange(len(df), dtype=np.int64)
    return df


def load_matches_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    return preprocess_matches_df(df)


def preprocess_stadiums_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "league" in df.columns:
        df["league"] = df["league"].astype(str)
    if "team_name" in df.columns:
        df["team_name"] = df["team_name"].astype(str)
    for c in ["latitude", "longitude"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "status" not in df.columns:
        df["status"] = "success"
    return df


def load_stadiums_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return preprocess_stadiums_df(df)


def add_implied_probability_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    odds = df[_ODDS_COLS].copy()
    for c in _ODDS_COLS:
        odds.loc[odds[c] <= 0, c] = np.nan
    q_home = 1.0 / odds["home_odds"]
    q_draw = 1.0 / odds["draw_odds"]
    q_away = 1.0 / odds["away_odds"]
    s = q_home + q_draw + q_away
    df["overround"] = s - 1.0
    df["p_home"] = q_home / s
    df["p_draw"] = q_draw / s
    df["p_away"] = q_away / s
    return df


def build_team_match_long(matches: pd.DataFrame) -> pd.DataFrame:
    needed = ["match_id", "date", "div", "home_team", "away_team", "home_goals", "away_goals"]
    missing = [c for c in needed if c not in matches.columns]
    if missing:
        raise ValueError(f"matches df missing columns: {missing}")

    base = matches[needed].copy()
    home = base.rename(columns={
        "home_team": "team",
        "away_team": "opponent",
        "home_goals": "goals_for",
        "away_goals": "goals_against",
    })
    home["is_home"] = 1

    away = base.rename(columns={
        "away_team": "team",
        "home_team": "opponent",
        "away_goals": "goals_for",
        "home_goals": "goals_against",
    })
    away["is_home"] = 0

    long_df = pd.concat([home, away], ignore_index=True)
    long_df["points"] = np.where(
        long_df["goals_for"] > long_df["goals_against"], 3,
        np.where(long_df["goals_for"] == long_df["goals_against"], 1, 0),
    )
    long_df = long_df.sort_values(["div", "team", "date", "match_id"]).reset_index(drop=True)
    return long_df


def add_rolling_form_features(long_df: pd.DataFrame, windows: Iterable[int], ewm_span: int = 6) -> pd.DataFrame:
    long_df = long_df.copy()
    long_df = long_df.sort_values(["div", "team", "date", "match_id"]).reset_index(drop=True)

    g = long_df.groupby(["div", "team"], sort=False)
    g_loc = long_df.groupby(["div", "team", "is_home"], sort=False)

    long_df["team_matches_played"] = g.cumcount()
    long_df["team_loc_matches_played"] = g_loc.cumcount()
    long_df["rest_days"] = g["date"].diff().dt.days

    # compute rolling means for goals-for, goals-against, points, and low-score indicator
    long_df["low_score"] = ((long_df["goals_for"] + long_df["goals_against"]) <= 2).astype(int)

    for w in windows:
        for col, prefix in [("goals_for", "gf"), ("goals_against", "ga"), ("points", "pts"), ("low_score", "low_score")]:
            roll = g[col].rolling(window=int(w), min_periods=1).mean()
            long_df[f"{prefix}_mean_{w}"] = roll.groupby(level=[0, 1]).shift(1).reset_index(level=[0, 1], drop=True)

            roll_loc = g_loc[col].rolling(window=int(w), min_periods=1).mean()
            long_df[f"{prefix}_loc_mean_{w}"] = roll_loc.groupby(level=[0, 1, 2]).shift(1).reset_index(level=[0, 1, 2], drop=True)

    span = max(int(ewm_span), 2)
    for col, prefix in [("goals_for", "gf"), ("goals_against", "ga"), ("points", "pts"), ("low_score", "low_score")]:
        long_df[f"{prefix}_ewm"] = g[col].transform(lambda s: s.ewm(span=span, adjust=False, min_periods=1).mean().shift(1))
        long_df[f"{prefix}_loc_ewm"] = g_loc[col].transform(lambda s: s.ewm(span=span, adjust=False, min_periods=1).mean().shift(1))

    opp_base_cols = []
    for w in windows:
        opp_base_cols.extend([f"gf_mean_{w}", f"ga_mean_{w}", f"gf_loc_mean_{w}", f"ga_loc_mean_{w}", f"low_score_mean_{w}", f"low_score_loc_mean_{w}"])
    opp_base_cols.extend(["gf_ewm", "ga_ewm", "gf_loc_ewm", "ga_loc_ewm", "low_score_ewm", "low_score_loc_ewm"])
    opp_view = long_df[["match_id", "team"] + opp_base_cols].rename(
        columns={"team": "opponent", **{c: f"opp_{c}" for c in opp_base_cols}}
    )
    long_df = long_df.merge(opp_view, on=["match_id", "opponent"], how="left")

    for w in windows:
        long_df[f"adj_gf_mean_{w}"] = long_df[f"gf_mean_{w}"] - long_df[f"opp_ga_mean_{w}"]
        long_df[f"adj_ga_mean_{w}"] = long_df[f"ga_mean_{w}"] - long_df[f"opp_gf_mean_{w}"]
        long_df[f"adj_gf_loc_mean_{w}"] = long_df[f"gf_loc_mean_{w}"] - long_df[f"opp_ga_loc_mean_{w}"]
        long_df[f"adj_ga_loc_mean_{w}"] = long_df[f"ga_loc_mean_{w}"] - long_df[f"opp_gf_loc_mean_{w}"]

    long_df["adj_gf_ewm"] = long_df["gf_ewm"] - long_df["opp_ga_ewm"]
    long_df["adj_ga_ewm"] = long_df["ga_ewm"] - long_df["opp_gf_ewm"]
    long_df["adj_gf_loc_ewm"] = long_df["gf_loc_ewm"] - long_df["opp_ga_loc_ewm"]
    long_df["adj_ga_loc_ewm"] = long_df["ga_loc_ewm"] - long_df["opp_gf_loc_ewm"]

    long_df = long_df.drop(columns=[f"opp_{c}" for c in opp_base_cols], errors="ignore")
    return long_df


def merge_match_and_team_features(matches: pd.DataFrame, long_with_features: pd.DataFrame) -> pd.DataFrame:
    matches = matches.copy()
    feature_cols = [c for c in long_with_features.columns if c not in {
        "match_id", "date", "div", "team", "opponent", "is_home", "goals_for", "goals_against", "points"
    }]

    home_long = long_with_features[long_with_features["is_home"] == 1][["match_id"] + feature_cols].copy()
    away_long = long_with_features[long_with_features["is_home"] == 0][["match_id"] + feature_cols].copy()
    home_long = home_long.rename(columns={c: f"home_{c}" for c in feature_cols})
    away_long = away_long.rename(columns={c: f"away_{c}" for c in feature_cols})

    out = matches.merge(home_long, on="match_id", how="left")
    out = out.merge(away_long, on="match_id", how="left")
    return out


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(lat2.astype(float))
    lon2 = np.radians(lon2.astype(float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


def add_travel_distance_feature(matches: pd.DataFrame, stadiums: pd.DataFrame) -> pd.DataFrame:
    matches = matches.copy()
    if stadiums is None or stadiums.empty:
        matches["away_travel_km"] = np.nan
        return matches

    st = stadiums.copy()
    if "status" in st.columns:
        st = st[st["status"].astype(str).str.lower().eq("success")]
    keep_cols = [c for c in ["league", "team_name", "latitude", "longitude"] if c in st.columns]
    st = st[keep_cols].dropna(subset=[c for c in ["latitude", "longitude", "team_name"] if c in keep_cols]).copy()
    st = st.rename(columns={"league": "div"})

    h = st.rename(columns={
        "team_name": "home_team",
        "latitude": "home_lat",
        "longitude": "home_lon",
    })
    a = st.rename(columns={
        "team_name": "away_team",
        "latitude": "away_lat",
        "longitude": "away_lon",
    })

    on_cols = [c for c in ["div"] if c in h.columns and c in matches.columns]
    m = matches.merge(h[[*(on_cols), "home_team", "home_lat", "home_lon"]], on=[*(on_cols), "home_team"], how="left")
    m = m.merge(a[[*(on_cols), "away_team", "away_lat", "away_lon"]], on=[*(on_cols), "away_team"], how="left")

    ok = m[["home_lat", "home_lon", "away_lat", "away_lon"]].notna().all(axis=1)
    m["away_travel_km"] = np.nan
    if ok.any():
        m.loc[ok, "away_travel_km"] = _haversine_km(
            m.loc[ok, "home_lat"], m.loc[ok, "home_lon"],
            m.loc[ok, "away_lat"], m.loc[ok, "away_lon"],
        )
    return m.drop(columns=[c for c in ["home_lat", "home_lon", "away_lat", "away_lon"] if c in m.columns], errors="ignore")


def add_matchup_difference_features(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    df = df.copy()
    for base in ["team_matches_played", "team_loc_matches_played", "rest_days"]:
        h = f"home_{base}"
        a = f"away_{base}"
        if h in df.columns and a in df.columns:
            df[f"diff_{base}"] = df[h] - df[a]

    for w in config.windows:
        for prefix in ["gf", "ga", "pts"]:
            for suffix in ["mean", "loc_mean"]:
                h = f"home_{prefix}_{suffix}_{w}"
                a = f"away_{prefix}_{suffix}_{w}"
                if h in df.columns and a in df.columns:
                    df[f"diff_{prefix}_{suffix}_{w}"] = df[h] - df[a]

    if config.use_ewm_features:
        for prefix in ["gf", "ga", "pts"]:
            for suffix in ["ewm", "loc_ewm"]:
                h = f"home_{prefix}_{suffix}"
                a = f"away_{prefix}_{suffix}"
                if h in df.columns and a in df.columns:
                    df[f"diff_{prefix}_{suffix}"] = df[h] - df[a]

    if config.use_adjusted_features:
        for w in config.windows:
            for prefix in ["adj_gf", "adj_ga"]:
                for suffix in ["mean", "loc_mean"]:
                    h = f"home_{prefix}_{suffix}_{w}"
                    a = f"away_{prefix}_{suffix}_{w}"
                    if h in df.columns and a in df.columns:
                        df[f"diff_{prefix}_{suffix}_{w}"] = df[h] - df[a]
        if config.use_ewm_features:
            for prefix in ["adj_gf", "adj_ga"]:
                for suffix in ["ewm", "loc_ewm"]:
                    h = f"home_{prefix}_{suffix}"
                    a = f"away_{prefix}_{suffix}"
                    if h in df.columns and a in df.columns:
                        df[f"diff_{prefix}_{suffix}"] = df[h] - df[a]
    return df


def model_categorical_columns() -> Tuple[str, ...]:
    return ("div", "home_team", "away_team")


def model_numeric_columns(config: FeatureConfig) -> Tuple[str, ...]:
    cols = ["p_home", "p_draw", "p_away", "overround"]
    if config.use_travel_distance:
        cols.append("away_travel_km")

    for side in ("home", "away"):
        cols.extend([f"{side}_team_matches_played", f"{side}_team_loc_matches_played", f"{side}_rest_days"])
        for w in config.windows:
            for prefix in ("gf", "ga", "pts", "low_score"):
                cols.append(f"{side}_{prefix}_mean_{w}")
                cols.append(f"{side}_{prefix}_loc_mean_{w}")
        if config.use_ewm_features:
            for prefix in ("gf", "ga", "pts", "low_score"):
                cols.append(f"{side}_{prefix}_ewm")
                cols.append(f"{side}_{prefix}_loc_ewm")
        if config.use_adjusted_features:
            for w in config.windows:
                for prefix in ("adj_gf", "adj_ga"):
                    cols.append(f"{side}_{prefix}_mean_{w}")
                    cols.append(f"{side}_{prefix}_loc_mean_{w}")
            if config.use_ewm_features:
                for prefix in ("adj_gf", "adj_ga"):
                    cols.append(f"{side}_{prefix}_ewm")
                    cols.append(f"{side}_{prefix}_loc_ewm")

    cols.extend(["diff_team_matches_played", "diff_team_loc_matches_played", "diff_rest_days"])
    for w in config.windows:
        for prefix in ("gf", "ga", "pts"):
            cols.append(f"diff_{prefix}_mean_{w}")
            cols.append(f"diff_{prefix}_loc_mean_{w}")
    if config.use_ewm_features:
        for prefix in ("gf", "ga", "pts"):
            cols.append(f"diff_{prefix}_ewm")
            cols.append(f"diff_{prefix}_loc_ewm")
    if config.use_adjusted_features:
        for w in config.windows:
            for prefix in ("adj_gf", "adj_ga"):
                cols.append(f"diff_{prefix}_mean_{w}")
                cols.append(f"diff_{prefix}_loc_mean_{w}")
        if config.use_ewm_features:
            for prefix in ("adj_gf", "adj_ga"):
                cols.append(f"diff_{prefix}_ewm")
                cols.append(f"diff_{prefix}_loc_ewm")

    return tuple(cols)


def build_training_frame(matches: pd.DataFrame, stadiums: Optional[pd.DataFrame], config: FeatureConfig) -> pd.DataFrame:
    df = matches.copy()
    df = add_implied_probability_features(df)
    if config.use_travel_distance:
        df = add_travel_distance_feature(df, stadiums if stadiums is not None else pd.DataFrame())

    long_df = build_team_match_long(df)
    long_df = add_rolling_form_features(long_df, windows=config.windows, ewm_span=int(config.ewm_span))
    df = merge_match_and_team_features(df, long_df)
    df = add_matchup_difference_features(df, config)
    return df


def _team_form_asof(
    long_df: pd.DataFrame,
    div: str,
    team: str,
    asof_date: pd.Timestamp,
    windows: Iterable[int],
    ewm_span: int,
    is_home_context: int,
) -> Dict[str, float]:
    hist_all = long_df[
        (long_df["div"] == div) &
        (long_df["team"] == team) &
        (long_df["date"] < asof_date)
    ].sort_values("date")
    hist_loc = hist_all[hist_all["is_home"] == int(is_home_context)]

    feats: Dict[str, float] = {}
    feats["team_matches_played"] = float(len(hist_all))
    feats["team_loc_matches_played"] = float(len(hist_loc))
    feats["rest_days"] = float((asof_date - hist_all["date"].iloc[-1]).days) if len(hist_all) else np.nan

    for w in windows:
        ta = hist_all.tail(int(w))
        tl = hist_loc.tail(int(w))
        feats[f"gf_mean_{w}"] = float(ta["goals_for"].mean()) if len(ta) else np.nan
        feats[f"ga_mean_{w}"] = float(ta["goals_against"].mean()) if len(ta) else np.nan
        feats[f"pts_mean_{w}"] = float(ta["points"].mean()) if len(ta) else np.nan
        feats[f"gf_loc_mean_{w}"] = float(tl["goals_for"].mean()) if len(tl) else np.nan
        feats[f"ga_loc_mean_{w}"] = float(tl["goals_against"].mean()) if len(tl) else np.nan
        feats[f"pts_loc_mean_{w}"] = float(tl["points"].mean()) if len(tl) else np.nan

    span = max(int(ewm_span), 2)
    if len(hist_all):
        feats["gf_ewm"] = float(hist_all["goals_for"].ewm(span=span, adjust=False, min_periods=1).mean().iloc[-1])
        feats["ga_ewm"] = float(hist_all["goals_against"].ewm(span=span, adjust=False, min_periods=1).mean().iloc[-1])
        feats["pts_ewm"] = float(hist_all["points"].ewm(span=span, adjust=False, min_periods=1).mean().iloc[-1])
    else:
        feats["gf_ewm"] = np.nan
        feats["ga_ewm"] = np.nan
        feats["pts_ewm"] = np.nan

    if len(hist_loc):
        feats["gf_loc_ewm"] = float(hist_loc["goals_for"].ewm(span=span, adjust=False, min_periods=1).mean().iloc[-1])
        feats["ga_loc_ewm"] = float(hist_loc["goals_against"].ewm(span=span, adjust=False, min_periods=1).mean().iloc[-1])
        feats["pts_loc_ewm"] = float(hist_loc["points"].ewm(span=span, adjust=False, min_periods=1).mean().iloc[-1])
    else:
        feats["gf_loc_ewm"] = np.nan
        feats["ga_loc_ewm"] = np.nan
        feats["pts_loc_ewm"] = np.nan

    return feats


def build_single_match_features(
    *,
    div: str,
    home_team: str,
    away_team: str,
    asof_date: pd.Timestamp,
    home_odds: Optional[float],
    draw_odds: Optional[float],
    away_odds: Optional[float],
    long_df: pd.DataFrame,
    stadiums: Optional[pd.DataFrame],
    config: FeatureConfig,
) -> pd.DataFrame:
    row: Dict[str, object] = {
        "div": str(div),
        "home_team": str(home_team),
        "away_team": str(away_team),
        "home_odds": float(home_odds) if home_odds is not None else np.nan,
        "draw_odds": float(draw_odds) if draw_odds is not None else np.nan,
        "away_odds": float(away_odds) if away_odds is not None else np.nan,
    }
    tmp = pd.DataFrame([row])
    tmp = add_implied_probability_features(tmp)
    row.update({
        "p_home": float(tmp.loc[0, "p_home"]) if "p_home" in tmp else np.nan,
        "p_draw": float(tmp.loc[0, "p_draw"]) if "p_draw" in tmp else np.nan,
        "p_away": float(tmp.loc[0, "p_away"]) if "p_away" in tmp else np.nan,
        "overround": float(tmp.loc[0, "overround"]) if "overround" in tmp else np.nan,
    })

    h_feats = _team_form_asof(long_df, div=str(div), team=str(home_team), asof_date=asof_date,
                              windows=config.windows, ewm_span=int(config.ewm_span), is_home_context=1)
    a_feats = _team_form_asof(long_df, div=str(div), team=str(away_team), asof_date=asof_date,
                              windows=config.windows, ewm_span=int(config.ewm_span), is_home_context=0)
    for k, v in h_feats.items():
        row[f"home_{k}"] = v
    for k, v in a_feats.items():
        row[f"away_{k}"] = v

    for w in config.windows:
        row[f"home_adj_gf_mean_{w}"] = row.get(f"home_gf_mean_{w}", np.nan) - row.get(f"away_ga_mean_{w}", np.nan)
        row[f"home_adj_ga_mean_{w}"] = row.get(f"home_ga_mean_{w}", np.nan) - row.get(f"away_gf_mean_{w}", np.nan)
        row[f"away_adj_gf_mean_{w}"] = row.get(f"away_gf_mean_{w}", np.nan) - row.get(f"home_ga_mean_{w}", np.nan)
        row[f"away_adj_ga_mean_{w}"] = row.get(f"away_ga_mean_{w}", np.nan) - row.get(f"home_gf_mean_{w}", np.nan)
        row[f"home_adj_gf_loc_mean_{w}"] = row.get(f"home_gf_loc_mean_{w}", np.nan) - row.get(f"away_ga_loc_mean_{w}", np.nan)
        row[f"home_adj_ga_loc_mean_{w}"] = row.get(f"home_ga_loc_mean_{w}", np.nan) - row.get(f"away_gf_loc_mean_{w}", np.nan)
        row[f"away_adj_gf_loc_mean_{w}"] = row.get(f"away_gf_loc_mean_{w}", np.nan) - row.get(f"home_ga_loc_mean_{w}", np.nan)
        row[f"away_adj_ga_loc_mean_{w}"] = row.get(f"away_ga_loc_mean_{w}", np.nan) - row.get(f"home_gf_loc_mean_{w}", np.nan)

    row["home_adj_gf_ewm"] = row.get("home_gf_ewm", np.nan) - row.get("away_ga_ewm", np.nan)
    row["home_adj_ga_ewm"] = row.get("home_ga_ewm", np.nan) - row.get("away_gf_ewm", np.nan)
    row["away_adj_gf_ewm"] = row.get("away_gf_ewm", np.nan) - row.get("home_ga_ewm", np.nan)
    row["away_adj_ga_ewm"] = row.get("away_ga_ewm", np.nan) - row.get("home_gf_ewm", np.nan)
    row["home_adj_gf_loc_ewm"] = row.get("home_gf_loc_ewm", np.nan) - row.get("away_ga_loc_ewm", np.nan)
    row["home_adj_ga_loc_ewm"] = row.get("home_ga_loc_ewm", np.nan) - row.get("away_gf_loc_ewm", np.nan)
    row["away_adj_gf_loc_ewm"] = row.get("away_gf_loc_ewm", np.nan) - row.get("home_ga_loc_ewm", np.nan)
    row["away_adj_ga_loc_ewm"] = row.get("away_ga_loc_ewm", np.nan) - row.get("home_gf_loc_ewm", np.nan)

    if config.use_travel_distance:
        try:
            tmp_match = pd.DataFrame([{"div": str(div), "home_team": str(home_team), "away_team": str(away_team)}])
            tmp_match = add_travel_distance_feature(tmp_match, stadiums if stadiums is not None else pd.DataFrame())
            row["away_travel_km"] = float(tmp_match.loc[0, "away_travel_km"])
        except Exception:
            row["away_travel_km"] = np.nan

    df_row = pd.DataFrame([row])
    df_row = add_matchup_difference_features(df_row, config)
    return df_row
