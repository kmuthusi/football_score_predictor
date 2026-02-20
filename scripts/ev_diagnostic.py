"""ev_diagnostic.py

Compute model EV distribution across historical matches and report
recommendations for sensible defaults (ev_threshold, min_bankroll, min_stake).

Usage:
  python scripts/ev_diagnostic.py --artifact models/score_models.joblib --matches data/spi_matches.csv --stadiums data/stadium_coordinates.csv --recent-days 365

Outputs a short JSON summary to stdout.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from features import FeatureConfig, build_training_frame, load_matches_csv, load_stadiums_csv
from predict import load_artifact, predict_expected_goals, scoreline_probability_matrix, calibrate_scoreline_matrix


def summarize_ev_distribution(artifact_path: str, matches_path: str, stadiums_path: str | None = None, *, recent_days: int | None = 365) -> Dict[str, Any]:
    artifact = load_artifact(artifact_path)
    matches = load_matches_csv(matches_path)
    stadiums = load_stadiums_csv(stadiums_path) if stadiums_path else None

    cfg = FeatureConfig(**artifact.get("config", {}))

    frame = build_training_frame(matches, stadiums, cfg)
    frame = frame.dropna(subset=["home_goals", "away_goals"]).copy()
    frame = frame.sort_values("date").reset_index(drop=True)

    if recent_days is not None:
        cutoff = pd.Timestamp(frame["date"].max()) - pd.Timedelta(days=int(recent_days))
        sel = frame[frame["date"] >= cutoff].copy()
    else:
        sel = frame.copy()

    if len(sel) == 0:
        raise RuntimeError("No matches selected for EV diagnostic.")

    cat_cols = list(artifact.get("cat_cols", []))
    num_cols = list(artifact.get("num_cols", []))
    X = sel[cat_cols + num_cols].copy()

    lam_h, lam_a = predict_expected_goals(artifact, X)

    max_goals = int(artifact.get("config", {}).get("max_goals", 6))
    cal_cfg = artifact.get("scoreline_calibration", None)

    evs: List[float] = []
    ev_by_side = {"home": [], "draw": [], "away": []}
    outcomes = []

    for idx, r in enumerate(sel.itertuples()):
        lh = float(lam_h[idx])
        la = float(lam_a[idx])
        div = str(getattr(r, "div")) if getattr(r, "div") is not None else None

        mat = scoreline_probability_matrix(lh, la, max_goals=max_goals, include_tail_bucket=True, rho=None)
        mat = calibrate_scoreline_matrix(mat, cal_cfg, div=div)

        # compute W/D/L probs
        arr = mat.to_numpy(dtype=float)
        p_home = float(np.tril(arr, k=-1).sum())
        p_draw = float(np.trace(arr))
        p_away = float(np.triu(arr, k=1).sum())

        ho = float(getattr(r, "home_odds") or math.nan)
        do = float(getattr(r, "draw_odds") or math.nan)
        ao = float(getattr(r, "away_odds") or math.nan)

        # implied probabilities (vig-adjusted)
        if np.isfinite(ho) and np.isfinite(do) and np.isfinite(ao):
            qh = 1.0 / max(ho, 1e-12)
            qd = 1.0 / max(do, 1e-12)
            qa = 1.0 / max(ao, 1e-12)
            s = qh + qd + qa
            imp_h, imp_d, imp_a = (qh / s, qd / s, qa / s)
        else:
            imp_h, imp_d, imp_a = (math.nan, math.nan, math.nan)

        ev_h = p_home * ho - 1.0 if np.isfinite(ho) else float("-inf")
        ev_d = p_draw * do - 1.0 if np.isfinite(do) else float("-inf")
        ev_a = p_away * ao - 1.0 if np.isfinite(ao) else float("-inf")

        evs.append(max(ev_h, ev_d, ev_a))
        ev_by_side["home"].append(ev_h)
        ev_by_side["draw"].append(ev_d)
        ev_by_side["away"].append(ev_a)

        # real outcome
        yh = int(getattr(r, "home_goals"))
        ya = int(getattr(r, "away_goals"))
        true = 0 if yh > ya else (1 if yh == ya else 2)
        outcomes.append(true)

    evs_arr = np.asarray([e for e in evs if np.isfinite(e)], dtype=float)

    def pct(x):
        return float(np.nanpercentile(evs_arr, x)) if evs_arr.size else float("nan")

    # Simple rule performance: stake 1 unit on side with max EV when EV > threshold
    def rule_stats(threshold: float) -> Dict[str, float]:
        total_bets = 0
        total_pnl = 0.0
        for i, r in enumerate(sel.itertuples()):
            evs_i = [ev_by_side["home"][i], ev_by_side["draw"][i], ev_by_side["away"][i]]
            best = int(np.argmax(evs_i))
            if evs_i[best] > threshold:
                total_bets += 1
                # outcome
                true = outcomes[i]
                if best == true:
                    odds = [getattr(r, "home_odds"), getattr(r, "draw_odds"), getattr(r, "away_odds")][best]
                    total_pnl += float(odds - 1.0)
                else:
                    total_pnl -= 1.0
        roi_on_staked = float(total_pnl / total_bets) if total_bets > 0 else float("nan")
        prop_bet = float(total_bets) / float(len(sel))
        return {"threshold": threshold, "n_bets": total_bets, "prop_bet": prop_bet, "roi_on_staked": roi_on_staked}

    thresholds = [0.0, 0.005, 0.01, 0.02, 0.05]
    rules = [rule_stats(t) for t in thresholds]

    out = {
        "checked_matches": int(len(sel)),
        "prop_matches_with_positive_ev": float((np.asarray(evs) > 0.0).sum()) / float(len(evs)),
        "ev_percentiles": {"p50": pct(50), "p75": pct(75), "p90": pct(90), "p95": pct(95)},
        "rule_sim": rules,
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", required=True)
    p.add_argument("--matches", required=True)
    p.add_argument("--stadiums", default=None)
    p.add_argument("--recent-days", type=int, default=365)
    args = p.parse_args()

    rpt = summarize_ev_distribution(args.artifact, args.matches, args.stadiums, recent_days=args.recent_days)
    print(json.dumps(rpt, indent=2))


if __name__ == "__main__":
    main()
