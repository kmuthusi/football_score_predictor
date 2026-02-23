# rl_env.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from features import FeatureConfig, build_training_frame, load_matches_csv, load_stadiums_csv
from predict import (
    load_artifact,
    predict_expected_goals,
    scoreline_probability_matrix,
    calibrate_scoreline_matrix,
)

# --- small utility copied from your Streamlit logic ---
def wdl_from_scoreline_matrix(mat: pd.DataFrame) -> Tuple[float, float, float]:
    arr = mat.to_numpy(dtype=float)
    p_home = float(np.tril(arr, k=-1).sum())
    p_draw = float(np.trace(arr))
    p_away = float(np.triu(arr, k=1).sum())
    return p_home, p_draw, p_away


def implied_probs_from_odds(home_odds: float, draw_odds: float, away_odds: float) -> Tuple[float, float, float]:
    # Matches features.add_implied_probability_features() logic
    qh = 1.0 / max(home_odds, 1e-12)
    qd = 1.0 / max(draw_odds, 1e-12)
    qa = 1.0 / max(away_odds, 1e-12)
    s = qh + qd + qa
    if s <= 0:
        return (1/3, 1/3, 1/3)
    return qh / s, qd / s, qa / s


@dataclass
class EnvConfig:
    initial_bankroll: float = 1000.0
    max_stake_frac: float = 0.05   # 5% max stake per match
    stake_fracs: Tuple[float, ...] = (0.0, 0.01, 0.02, 0.05)  # 0 = skip sizing
    # reward_mode: "profit" or "log_growth"
    reward_mode: str = "log_growth"
    # per-unit penalty proportional to predicted probability of a low-scoring game
    low_score_penalty: float = 0.0


class FootballBettingEnv:
    """
    One step = one historical match (chronological).
    Action = discrete (outcome_choice, stake_frac_idx).
    """

    def __init__(self, artifact_path: str, matches_path: str, stadiums_path: Optional[str], *,
                 split: str = "train", env_cfg: Optional[EnvConfig] = None):
        self.env_cfg = env_cfg or EnvConfig()

        self.artifact = load_artifact(artifact_path)
        self.matches = load_matches_csv(matches_path)
        self.stadiums = load_stadiums_csv(stadiums_path) if stadiums_path else None

        # Use the exact feature config stored in the artifact
        cfg = FeatureConfig(**self.artifact.get("config", {}))

        # Precompute per-match feature frame once (fast + leakage-safe)
        frame = build_training_frame(self.matches, self.stadiums, cfg)
        frame = frame.dropna(subset=["home_goals", "away_goals"]).copy()
        frame = frame.sort_values("date").reset_index(drop=True)

        # Reuse artifact split convention
        cutoff = pd.Timestamp(self.artifact["cutoff_date"])
        if split == "train":
            self.df = frame[frame["date"] < cutoff].reset_index(drop=True)
        elif split == "test":
            self.df = frame[frame["date"] >= cutoff].reset_index(drop=True)
        else:
            raise ValueError("split must be 'train' or 'test'")

        # Model inputs
        self.cat_cols = list(self.artifact.get("cat_cols", []))
        self.num_cols = list(self.artifact.get("num_cols", []))
        X = self.df[self.cat_cols + self.num_cols].copy()

        # Precompute lambdas for speed
        lam_h, lam_a = predict_expected_goals(self.artifact, X)
        self.df["_lam_home"] = lam_h
        self.df["_lam_away"] = lam_a

        # DC + calibration configs from artifact (same keys used in app.py)
        self.scoreline_cal_cfg = self.artifact.get("scoreline_calibration", None)
        dc_cfg = self.artifact.get("dixon_coles", {}) or {}
        self.rho_global = dc_cfg.get("rho_global", self.artifact.get("dixon_coles_rho"))
        self.rho_by_div = dc_cfg.get("rho_by_div", {}) or {}

        # Episode state
        self.t = 0
        self.bankroll = float(self.env_cfg.initial_bankroll)

        # Action mapping
        # action_id = outcome_idx * len(stake_fracs) + stake_idx
        # outcome_idx: 0=skip, 1=home, 2=draw, 3=away
        self.n_stakes = len(self.env_cfg.stake_fracs)
        self.n_actions = 4 * self.n_stakes

    def reset(self) -> np.ndarray:
        self.t = 0
        self.bankroll = float(self.env_cfg.initial_bankroll)
        return self._obs(self.t)

    def _rho_for_row(self, div: str) -> Optional[float]:
        if div is None:
            return float(self.rho_global) if self.rho_global is not None else None
        if str(div) in self.rho_by_div:
            return float(self.rho_by_div[str(div)])
        return float(self.rho_global) if self.rho_global is not None else None

    def _obs(self, idx: int) -> np.ndarray:
        r = self.df.iloc[idx]
        div = str(r["div"])

        lam_h = float(r["_lam_home"])
        lam_a = float(r["_lam_away"])
        rho = self._rho_for_row(div)

        mat = scoreline_probability_matrix(
            lam_h, lam_a,
            max_goals=int(self.artifact["config"]["max_goals"]),
            include_tail_bucket=True,
            rho=rho,
        )
        mat = calibrate_scoreline_matrix(mat, self.scoreline_cal_cfg, div=div)
        arr = mat.to_numpy(dtype=float)

        pH, pD, pA = wdl_from_scoreline_matrix(mat)

        # Odds + implied probs (vig-adjusted)
        ho = float(r["home_odds"]) if pd.notna(r["home_odds"]) else np.nan
        do = float(r["draw_odds"]) if pd.notna(r["draw_odds"]) else np.nan
        ao = float(r["away_odds"]) if pd.notna(r["away_odds"]) else np.nan

        # If odds missing, treat as "no trade"
        if not np.isfinite(ho) or not np.isfinite(do) or not np.isfinite(ao):
            impH, impD, impA = (1/3, 1/3, 1/3)
        else:
            impH, impD, impA = implied_probs_from_odds(ho, do, ao)

        edgeH, edgeD, edgeA = (pH - impH), (pD - impD), (pA - impA)

        # compute low-score probability (total goals <= 1)
        low1 = float(arr[0,0] + arr[0,1] + arr[1,0]) if arr.shape[0] > 1 else float(arr[0,0])

        # build observation vector matching the layout used during RL training
        obs = np.array([
            pH, pD, pA,
            impH, impD, impA,            # implied probabilities
            edgeH, edgeD, edgeA,
            ho if np.isfinite(ho) else 0.0,
            do if np.isfinite(do) else 0.0,
            ao if np.isfinite(ao) else 0.0,
            np.log(max(self.bankroll, 1e-12)),
            low1,
        ], dtype=np.float32)
        return obs

    def step(self, action_id: int) -> Tuple[np.ndarray, float, bool, Dict]:
        r = self.df.iloc[self.t]

        outcome_idx = int(action_id // self.n_stakes)  # 0..3
        stake_idx = int(action_id % self.n_stakes)
        stake_frac = float(self.env_cfg.stake_fracs[stake_idx])

        # decode actual match outcome
        yh = int(r["home_goals"])
        ya = int(r["away_goals"])
        if yh > ya:
            true = "home"
        elif yh == ya:
            true = "draw"
        else:
            true = "away"

        # odds
        ho = float(r["home_odds"]) if pd.notna(r["home_odds"]) else np.nan
        do = float(r["draw_odds"]) if pd.notna(r["draw_odds"]) else np.nan
        ao = float(r["away_odds"]) if pd.notna(r["away_odds"]) else np.nan

        bankroll_before = self.bankroll
        reward = 0.0
        info = {"bankroll_before": bankroll_before}

        # If skip or odds missing -> no bet
        if outcome_idx == 0 or (not np.isfinite(ho) or not np.isfinite(do) or not np.isfinite(ao)):
            pass
        else:
            stake_frac = min(stake_frac, float(self.env_cfg.max_stake_frac))
            stake = bankroll_before * stake_frac

            pick = {1: "home", 2: "draw", 3: "away"}[outcome_idx]
            odds = {"home": ho, "draw": do, "away": ao}[pick]

            if pick == true:
                pnl = stake * (odds - 1.0)
            else:
                pnl = -stake

            self.bankroll = max(0.0, bankroll_before + pnl)

            if self.env_cfg.reward_mode == "profit":
                reward = float(pnl)
            else:
                # log growth is more stable + discourages ruin
                reward = float(np.log(max(self.bankroll, 1e-12)) - np.log(max(bankroll_before, 1e-12)))

            info.update({"pick": pick, "stake": stake, "pnl": pnl})

        self.t += 1
        done = (self.t >= len(self.df)) or (self.bankroll <= 0.0)

        info["bankroll_after"] = self.bankroll
        info["t"] = self.t

        obs = self._obs(self.t) if not done else np.zeros(13, dtype=np.float32)
        # apply optional low-score penalty to reward
        if not done and self.env_cfg.low_score_penalty:
            # recompute low1 for current match (could reuse obs last element)
            low1 = obs[-1] if obs.size >= 13 else 0.0
            reward -= float(self.env_cfg.low_score_penalty) * float(low1)
        return obs, reward, done, info
