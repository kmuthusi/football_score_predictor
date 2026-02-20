"""
rl_eval.py

Greedy backtest evaluator for a trained RL policy (linear softmax REINFORCE policy).

Loads:
  - policy: models/rl_policy.joblib (joblib payload with W, b, obs_dim, train_cfg, optional obs_norm)
  - artifact: models/score_models.joblib
  - matches: data/spi_matches.csv
  - stadiums: data/stadium_coordinates.csv

Runs:
  - builds leakage-safe historical feature frame
  - uses scoreline model + calibration to compute W/D/L probs
  - greedy policy selects action in {skip, bet_home, bet_draw, bet_away}
  - simulates bankroll and prints ROI, max drawdown, bet count, etc.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from config import DEFAULT_ARTIFACT_ABS_PATH, DEFAULT_MATCHES_ABS_PATH, DEFAULT_STADIUMS_ABS_PATH
from features import FeatureConfig, build_training_frame, load_matches_csv, load_stadiums_csv
from predict import (
    load_artifact,
    predict_expected_goals,
    scoreline_probability_matrix,
    calibrate_scoreline_matrix,
)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    e = np.exp(z)
    s = float(np.sum(e))
    if s <= 0.0 or not np.isfinite(s):
        return np.full_like(logits, 1.0 / max(len(logits), 1), dtype=float)
    return e / s


def _wdl_from_scoreline_matrix(mat: pd.DataFrame) -> Tuple[float, float, float]:
    # Same logic as the Streamlit app: sum below-diagonal / diagonal / above-diagonal. :contentReference[oaicite:3]{index=3}
    arr = mat.to_numpy(dtype=float)
    p_home = float(np.tril(arr, k=-1).sum())
    p_draw = float(np.trace(arr))
    p_away = float(np.triu(arr, k=1).sum())
    return p_home, p_draw, p_away


def _resolve_test_split(frame: pd.DataFrame, artifact: Dict[str, Any], test_days_fallback: int) -> pd.DataFrame:
    # Same split convention used across your pipeline: artifact cutoff_date if present, else last N days fallback. :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}
    if artifact.get("cutoff_date"):
        cutoff = pd.Timestamp(str(artifact["cutoff_date"]))
        return frame[frame["date"] >= cutoff].copy()

    max_date = pd.Timestamp(frame["date"].max())
    cutoff = max_date - pd.Timedelta(days=int(test_days_fallback))
    return frame[frame["date"] >= cutoff].copy()


def _rho_for_div(div: str, artifact: Dict[str, Any]) -> Optional[float]:
    dc_cfg = artifact.get("dixon_coles", {}) or {}
    rho_global = dc_cfg.get("rho_global", artifact.get("dixon_coles_rho"))
    rho_by_div = dc_cfg.get("rho_by_div", {}) or {}
    if div is not None and str(div) in rho_by_div:
        return float(rho_by_div[str(div)])
    return float(rho_global) if rho_global is not None else None


def _max_drawdown(equity: np.ndarray) -> float:
    peak = -float("inf")
    mdd = 0.0
    for v in equity:
        peak = max(peak, float(v))
        if peak > 0:
            dd = (peak - float(v)) / peak
            mdd = max(mdd, dd)
    return float(mdd)


def main() -> None:
    p = argparse.ArgumentParser(description="Greedy backtest evaluator for RL betting policy")
    p.add_argument("--policy", type=str, default="models/rl_policy.joblib")
    p.add_argument("--artifact", type=str, default=DEFAULT_ARTIFACT_ABS_PATH)
    p.add_argument("--matches", type=str, default=DEFAULT_MATCHES_ABS_PATH)
    p.add_argument("--stadiums", type=str, default=DEFAULT_STADIUMS_ABS_PATH)

    p.add_argument("--test-days", type=int, default=365, help="Used only if artifact has no cutoff_date.")
    p.add_argument("--initial-bankroll", type=float, default=1000.0)
    p.add_argument("--stake-frac", type=float, default=0.01)
    p.add_argument("--max-stake-frac", type=float, default=0.05)

    p.add_argument("--save-bets", type=str, default="", help="Optional path to save per-bet log CSV.")
    args = p.parse_args()

    policy_path = Path(args.policy)
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy not found: {policy_path}")

    artifact_path = Path(args.artifact)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")

    policy = joblib.load(policy_path)
    W = np.asarray(policy["W"], dtype=float)
    b = np.asarray(policy["b"], dtype=float)

    if W.ndim != 2 or b.ndim != 1 or W.shape[0] != 4 or b.shape[0] != 4:
        raise ValueError(f"Unexpected policy shapes: W={W.shape}, b={b.shape} (expected W=(4,obs_dim), b=(4,))")

    obs_dim = int(W.shape[1])

    # Optional normalization (only if you saved it from training)
    obs_norm = policy.get("obs_norm", None)
    norm_mean = None
    norm_std = None
    if isinstance(obs_norm, dict) and "mean" in obs_norm and "std" in obs_norm:
        norm_mean = np.asarray(obs_norm["mean"], dtype=float).reshape(-1)
        norm_std = np.asarray(obs_norm["std"], dtype=float).reshape(-1)
        if norm_mean.shape[0] != obs_dim or norm_std.shape[0] != obs_dim:
            print("[warn] obs_norm stats exist but dim mismatch; ignoring normalization.")
            norm_mean, norm_std = None, None
    else:
        # If training used obs norm but you didn't save stats, results can differ.
        train_cfg = policy.get("train_cfg", {}) or {}
        if bool(train_cfg.get("use_obs_norm", False)):
            print("[warn] Policy was trained with obs normalization but no obs_norm stats were saved.")
            print("       Eval will run WITHOUT normalization (numbers may differ).")
            print("       Fix: save mean/std in rl_train.py when dumping the policy, or train with --no-obs-norm.")

    artifact = load_artifact(artifact_path)

    matches = load_matches_csv(args.matches)
    stadiums = load_stadiums_csv(args.stadiums) if args.stadiums else None

    cfg = FeatureConfig(**artifact.get("config", {}))
    frame = build_training_frame(matches, stadiums, cfg)
    frame = frame.dropna(subset=["home_goals", "away_goals"]).copy()
    frame = frame.sort_values("date").reset_index(drop=True)

    test_df = _resolve_test_split(frame, artifact, test_days_fallback=int(args.test_days))
    test_df = test_df.sort_values("date").reset_index(drop=True)
    if len(test_df) == 0:
        raise RuntimeError("Resolved test split is empty; cannot backtest.")

    # Model inputs consistent with artifact contract
    cat_cols = list(artifact.get("cat_cols", []))
    num_cols = list(artifact.get("num_cols", []))
    X_test = test_df[cat_cols + num_cols].copy()

    lam_h, lam_a = predict_expected_goals(artifact, X_test)

    max_goals = int(artifact.get("config", {}).get("max_goals", 6))
    cal_cfg = artifact.get("scoreline_calibration", None)

    # Backtest loop
    bankroll = float(args.initial_bankroll)
    eps = 1e-12

    equity = [bankroll]
    bet_rows = []

    total_staked = 0.0
    total_pnl = 0.0
    n_bets = 0
    n_wins = 0

    for i in range(len(test_df)):
        r = test_df.iloc[i]
        div = str(r["div"])

        ho = float(r["home_odds"]) if pd.notna(r["home_odds"]) else np.nan
        do = float(r["draw_odds"]) if pd.notna(r["draw_odds"]) else np.nan
        ao = float(r["away_odds"]) if pd.notna(r["away_odds"]) else np.nan

        impH = float(r["p_home"]) if pd.notna(r["p_home"]) else 0.0
        impD = float(r["p_draw"]) if pd.notna(r["p_draw"]) else 0.0
        impA = float(r["p_away"]) if pd.notna(r["p_away"]) else 0.0

        rho = _rho_for_div(div, artifact)

        mat = scoreline_probability_matrix(
            float(lam_h[i]),
            float(lam_a[i]),
            max_goals=max_goals,
            include_tail_bucket=True,
            rho=float(rho) if rho is not None else None,
        )
        mat = calibrate_scoreline_matrix(mat, cal_cfg, div=div)
        pH, pD, pA = _wdl_from_scoreline_matrix(mat)

        edgeH, edgeD, edgeA = (pH - impH), (pD - impD), (pA - impA)

        x = np.array(
            [
                pH, pD, pA,
                impH, impD, impA,
                edgeH, edgeD, edgeA,
                ho if np.isfinite(ho) else 0.0,
                do if np.isfinite(do) else 0.0,
                ao if np.isfinite(ao) else 0.0,
                math.log(max(bankroll, eps)),
            ],
            dtype=float,
        )
        if x.shape[0] != obs_dim:
            raise RuntimeError(f"obs_dim mismatch: policy expects {obs_dim}, eval built {x.shape[0]}")

        if norm_mean is not None and norm_std is not None:
            x_in = (x - norm_mean) / np.maximum(norm_std, 1e-8)
        else:
            x_in = x

        logits = W @ x_in + b

        # Validity mask: skip always valid; bet requires finite odds > 1
        valid = np.array(
            [
                True,
                bool(np.isfinite(ho) and ho > 1.0),
                bool(np.isfinite(do) and do > 1.0),
                bool(np.isfinite(ao) and ao > 1.0),
            ],
            dtype=bool,
        )
        masked_logits = np.where(valid, logits, -1e9)
        probs = _softmax(masked_logits)

        action = int(np.argmax(probs))  # greedy

        # Determine true outcome
        yh = int(r["home_goals"])
        ya = int(r["away_goals"])
        true = 0 if yh > ya else (1 if yh == ya else 2)  # 0=home,1=draw,2=away

        bankroll_before = bankroll
        stake = 0.0
        pnl = 0.0
        pick_name = "skip"

        if action == 0:
            pass
        else:
            stake_frac = min(float(args.stake_frac), float(args.max_stake_frac))
            stake = bankroll_before * stake_frac

            pick = action - 1  # 0=home,1=draw,2=away
            pick_name = ["home", "draw", "away"][pick]
            odds = float([ho, do, ao][pick])

            n_bets += 1
            total_staked += float(stake)

            if pick == true:
                pnl = stake * (odds - 1.0)
                n_wins += 1
            else:
                pnl = -stake

            bankroll = max(0.0, bankroll_before + pnl)
            total_pnl += float(pnl)

            bet_rows.append(
                {
                    "date": str(pd.Timestamp(r["date"]).date()),
                    "div": div,
                    "home_team": r["home_team"],
                    "away_team": r["away_team"],
                    "action": pick_name,
                    "stake": float(stake),
                    "odds": float(odds),
                    "pnl": float(pnl),
                    "bankroll_before": float(bankroll_before),
                    "bankroll_after": float(bankroll),
                    "pH": float(pH),
                    "pD": float(pD),
                    "pA": float(pA),
                    "impH": float(impH),
                    "impD": float(impD),
                    "impA": float(impA),
                }
            )

        equity.append(bankroll)

        if bankroll <= 0.0:
            break

    equity_arr = np.asarray(equity, dtype=float)
    final_bankroll = float(equity_arr[-1])
    roi_bankroll = (final_bankroll - float(args.initial_bankroll)) / float(args.initial_bankroll)
    roi_staked = (total_pnl / total_staked) if total_staked > 0 else float("nan")
    mdd = _max_drawdown(equity_arr)

    # Reporting
    test_start = str(pd.Timestamp(test_df["date"].min()).date())
    test_end = str(pd.Timestamp(test_df["date"].max()).date())
    cutoff = artifact.get("cutoff_date")

    print("=== RL Policy Backtest (Greedy) ===")
    print(f"Policy:   {policy_path}")
    print(f"Artifact: {artifact_path}")
    print(f"Test:     {test_start} -> {test_end}  (artifact cutoff_date={cutoff})")
    print(f"Matches evaluated: {len(equity_arr) - 1:,}")
    print()
    print("Trading stats:")
    print(f"  Bets placed:     {n_bets:,}")
    print(f"  Win rate:        {n_wins / max(n_bets, 1):.2%}")
    print(f"  Total staked:    {total_staked:.2f}")
    print(f"  Total P&L:       {total_pnl:.2f}")
    print(f"  Final bankroll:  {final_bankroll:.2f}")
    print()
    print("Performance:")
    print(f"  ROI (bankroll):  {roi_bankroll:.2%}")
    print(f"  ROI (on staked): {roi_staked:.2%}" if np.isfinite(roi_staked) else "  ROI (on staked): n/a (no bets)")
    print(f"  Max drawdown:    {mdd:.2%}")

    if args.save_bets:
        out_path = Path(args.save_bets)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(bet_rows).to_csv(out_path, index=False)
        print(f"\nSaved bet log to: {out_path}")


if __name__ == "__main__":
    main()
