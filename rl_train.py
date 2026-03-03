"""
rl_train.py

Minimal policy-gradient (REINFORCE) trainer for a simple betting policy on top of
the existing scoreline model artifact.

- Action space: 0=skip, 1=bet home, 2=bet draw, 3=bet away
- Stake: fixed fraction of current bankroll (clipped by max_stake_frac)
- Reward: log bankroll growth (default) or per-step profit

Example:
  python rl_train.py --artifact models/score_models.joblib --matches data/spi_matches.csv --stadiums data/stadium_coordinates_completed_full.csv --epochs 30

Notes:
- This script precomputes model W/D/L probabilities ONCE (expensive part),
  then trains cheaply over many epochs.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

from config import DEFAULT_ARTIFACT_ABS_PATH, DEFAULT_MATCHES_ABS_PATH, DEFAULT_STADIUMS_ABS_PATH
from features import FeatureConfig, build_training_frame, load_matches_csv, load_stadiums_csv
from predict import (
    load_artifact,
    predict_expected_goals,
    scoreline_probability_matrix,
    calibrate_scoreline_matrix,
)
from probability_utils import wdl_from_scoreline_matrix


# ----------------------------
# Small numeric helpers
# ----------------------------

def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    e = np.exp(z)
    s = float(np.sum(e))
    if s <= 0.0 or not np.isfinite(s):
        return np.full_like(logits, 1.0 / max(len(logits), 1), dtype=float)
    return e / s


def _discounted_cumsum(rewards: np.ndarray, gamma: float) -> np.ndarray:
    out = np.zeros_like(rewards, dtype=float)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = float(rewards[t]) + float(gamma) * running
        out[t] = running
    return out


class RunningNorm:
    """Tiny running mean/std normalizer for observations (helps stability)."""
    def __init__(self, dim: int, eps: float = 1e-8):
        self.dim = int(dim)
        self.eps = float(eps)
        self.n = 0
        self.mean = np.zeros(self.dim, dtype=float)
        self.M2 = np.zeros(self.dim, dtype=float)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=float)
        if x.shape != (self.dim,):
            raise ValueError(f"RunningNorm expected shape {(self.dim,)}, got {x.shape}")

        self.n += 1
        delta = x - self.mean
        self.mean += delta / float(self.n)
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def std(self) -> np.ndarray:
        if self.n < 2:
            return np.ones(self.dim, dtype=float)
        var = self.M2 / float(max(self.n - 1, 1))
        return np.sqrt(np.maximum(var, self.eps))

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=float) - self.mean) / self.std()


@dataclass
class TrainConfig:
    epochs: int = 30
    lr: float = 0.02
    gamma: float = 1.0
    seed: int = 7

    initial_bankroll: float = 1000.0
    stake_frac: float = 0.01
    max_stake_frac: float = 0.05
    min_bankroll: float = 10.0
    min_stake: float = 1.0
    ev_threshold: float = 0.01
    bet_penalty: float = 0.001
    # penalty applied proportional to predicted low-scoring probability
    low_score_penalty: float = 0.0

    reward_mode: str = "log_growth"  # "log_growth" or "profit"

    use_obs_norm: bool = True
    grad_clip_norm: float = 5.0
    eval_every: int = 5


# ----------------------------
# Data + model precompute
# ----------------------------

def _resolve_split(frame: pd.DataFrame, artifact: Dict[str, Any], *, test_days_fallback: int = 365) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if artifact.get("cutoff_date"):
        cutoff = pd.Timestamp(str(artifact["cutoff_date"]))
        train_df = frame[frame["date"] < cutoff].copy()
        test_df = frame[frame["date"] >= cutoff].copy()
        return train_df, test_df

    max_date = pd.Timestamp(frame["date"].max())
    cutoff = max_date - pd.Timedelta(days=int(test_days_fallback))
    train_df = frame[frame["date"] < cutoff].copy()
    test_df = frame[frame["date"] >= cutoff].copy()
    return train_df, test_df


def _rho_for_div(div: str, dixon_coles_cfg: Dict[str, Any]) -> Optional[float]:
    rho_global = dixon_coles_cfg.get("rho_global", None)
    rho_by_div = dixon_coles_cfg.get("rho_by_div", {}) or {}
    if div is not None and str(div) in rho_by_div:
        return float(rho_by_div[str(div)])
    if rho_global is not None:
        return float(rho_global)
    return None


def build_precomputed_arrays(
    *,
    artifact_path: str,
    matches_path: str,
    stadiums_path: Optional[str],
    split: str,
) -> Dict[str, np.ndarray]:
    """
    Precompute everything expensive ONCE:
      - features (build_training_frame)
      - lambdas (predict_expected_goals)
      - model W/D/L probs from calibrated scoreline matrix
    """
    artifact = load_artifact(artifact_path)
    cfg = FeatureConfig(**artifact.get("config", {}))

    matches = load_matches_csv(matches_path)
    stadiums = load_stadiums_csv(stadiums_path) if stadiums_path else None

    frame = build_training_frame(matches, stadiums, cfg)
    frame = frame.dropna(subset=["home_goals", "away_goals"]).copy()
    frame = frame.sort_values("date").reset_index(drop=True)

    train_df, test_df = _resolve_split(frame, artifact)
    if split == "train":
        df = train_df.reset_index(drop=True)
    elif split == "test":
        df = test_df.reset_index(drop=True)
    else:
        raise ValueError("split must be 'train' or 'test'")

    if len(df) == 0:
        raise RuntimeError(f"Resolved split '{split}' is empty; cannot train/eval.")

    cat_cols = list(artifact.get("cat_cols", []))
    num_cols = list(artifact.get("num_cols", []))
    needed = cat_cols + num_cols + ["home_odds", "draw_odds", "away_odds", "p_home", "p_draw", "p_away", "div", "home_goals", "away_goals"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in frame for RL wrapper: {missing}")

    X = df[cat_cols + num_cols].copy()
    lam_home, lam_away = predict_expected_goals(artifact, X)

    max_goals = int(artifact.get("config", {}).get("max_goals", 6))
    cal_cfg = artifact.get("scoreline_calibration", None)
    dc_cfg = artifact.get("dixon_coles", {}) or {}
    # backward-compat: if dixon_coles block absent, still try top-level rho
    if not dc_cfg:
        dc_cfg = {"rho_global": artifact.get("dixon_coles_rho"), "rho_by_div": {}}

    # Precompute W/D/L model probabilities (expensive loop, done once)
    pH = np.zeros(len(df), dtype=float)
    pD = np.zeros(len(df), dtype=float)
    pA = np.zeros(len(df), dtype=float)
    low1 = np.zeros(len(df), dtype=float)  # probability total goals <=1

    divs = df["div"].astype(str).to_numpy()
    for i in range(len(df)):
        div_i = str(divs[i])
        rho_i = _rho_for_div(div_i, dc_cfg)

        mat = scoreline_probability_matrix(
            float(lam_home[i]),
            float(lam_away[i]),
            max_goals=max_goals,
            include_tail_bucket=True,
            rho=float(rho_i) if rho_i is not None else None,
        )
        mat = calibrate_scoreline_matrix(mat, cal_cfg, div=div_i)
        arr = mat.to_numpy(dtype=float)
        ph, pd_, pa = wdl_from_scoreline_matrix(mat)
        pH[i], pD[i], pA[i] = ph, pd_, pa
        # compute low1 probability
        low1[i] = float(arr[0,0] + arr[0,1] + arr[1,0]) if arr.shape[0] > 1 else float(arr[0,0])

    # Market implied probs from odds (already computed in features.py)
    impH = df["p_home"].astype(float).to_numpy()
    impD = df["p_draw"].astype(float).to_numpy()
    impA = df["p_away"].astype(float).to_numpy()

    # Odds + outcomes
    ho = df["home_odds"].astype(float).to_numpy()
    do = df["draw_odds"].astype(float).to_numpy()
    ao = df["away_odds"].astype(float).to_numpy()

    yh = df["home_goals"].astype(int).to_numpy()
    ya = df["away_goals"].astype(int).to_numpy()
    out = np.zeros(len(df), dtype=np.int64)  # 0=home,1=draw,2=away
    out[yh == ya] = 1
    out[yh < ya] = 2

    return {
        "pH": pH, "pD": pD, "pA": pA, "low1": low1,
        "impH": impH, "impD": impD, "impA": impA,
        "ho": ho, "do": do, "ao": ao,
        "outcome": out,
    }


# ----------------------------
# Simulator (policy interacts with historical outcomes)
# ----------------------------

def run_episode(
    arrays: Dict[str, np.ndarray],
    *,
    W: np.ndarray,
    b: np.ndarray,
    rng: np.random.Generator,
    cfg: TrainConfig,
    greedy: bool = False,
    obs_norm: Optional[RunningNorm] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Returns (obs_mat, probs_mat, actions, stats)
      - obs_mat: [T, obs_dim] (normalized if cfg.use_obs_norm)
      - probs_mat: [T, n_actions]
      - actions: [T]
    """
    pH, pD, pA = arrays["pH"], arrays["pD"], arrays["pA"]
    low1_arr = arrays.get("low1", np.zeros(len(arrays.get("pH", [])), dtype=float))
    impH, impD, impA = arrays["impH"], arrays["impD"], arrays["impA"]
    ho, do, ao = arrays["ho"], arrays["do"], arrays["ao"]
    outcome = arrays["outcome"]

    n = len(outcome)
    n_actions = 4
    obs_dim = W.shape[1]

    bankroll = float(cfg.initial_bankroll)
    eps = 1e-12

    obs_mat = np.zeros((n, obs_dim), dtype=float)
    probs_mat = np.zeros((n, n_actions), dtype=float)
    actions = np.zeros(n, dtype=np.int64)
    rewards = np.zeros(n, dtype=float)

    n_bets = 0
    n_wins = 0
    total_pnl = 0.0

    for t in range(n):
        # Build observation (static parts + log bankroll)
        edgeH = float(pH[t] - impH[t]) if np.isfinite(impH[t]) else 0.0
        edgeD = float(pD[t] - impD[t]) if np.isfinite(impD[t]) else 0.0
        edgeA = float(pA[t] - impA[t]) if np.isfinite(impA[t]) else 0.0

        x = np.array(
            [
                float(pH[t]), float(pD[t]), float(pA[t]),
                float(impH[t]) if np.isfinite(impH[t]) else 0.0,
                float(impD[t]) if np.isfinite(impD[t]) else 0.0,
                float(impA[t]) if np.isfinite(impA[t]) else 0.0,
                edgeH, edgeD, edgeA,
                float(ho[t]) if np.isfinite(ho[t]) else 0.0,
                float(do[t]) if np.isfinite(do[t]) else 0.0,
                float(ao[t]) if np.isfinite(ao[t]) else 0.0,
                float(math.log(max(bankroll, eps))),
                float(low1_arr[t]),
            ],
            dtype=float,
        )

        if x.shape[0] != obs_dim:
            raise RuntimeError(f"obs_dim mismatch: got {x.shape[0]} but policy expects {obs_dim}")

        if cfg.use_obs_norm and obs_norm is not None:
            obs_norm.update(x)
            x_in = obs_norm.normalize(x)
        else:
            x_in = x

        # EV gating: treat bets as invalid unless model EV per unit exceeds threshold
        # (this does not prevent the policy from learning but reduces structural
        # exposure during episodes where no meaningful edge exists)
        # Note: validity applied when sampling actions via masked logits below.
        # apply low-score penalty to reward if configured

        logits = W @ x_in + b

        # Action validity mask:
        # - skip always valid
        # - bet actions valid only if corresponding odds are finite and > 1
        # EV gating: only consider bet actions when model EV per unit exceeds threshold
        ev_home = float(pH[t] * ho[t] - 1.0) if np.isfinite(ho[t]) else float("-inf")
        ev_draw = float(pD[t] * do[t] - 1.0) if np.isfinite(do[t]) else float("-inf")
        ev_away = float(pA[t] * ao[t] - 1.0) if np.isfinite(ao[t]) else float("-inf")

        valid = np.array(
            [
                True,
                bool(np.isfinite(ho[t]) and ho[t] > 1.0 and (ev_home > float(cfg.ev_threshold))),
                bool(np.isfinite(do[t]) and do[t] > 1.0 and (ev_draw > float(cfg.ev_threshold))),
                bool(np.isfinite(ao[t]) and ao[t] > 1.0 and (ev_away > float(cfg.ev_threshold))),
            ],
            dtype=bool,
        )
        masked_logits = np.where(valid, logits, -1e9)
        probs = _softmax(masked_logits)

        if greedy:
            a = int(np.argmax(probs))
        else:
            a = int(rng.choice(4, p=probs))

        bankroll_before = bankroll

        # Decode bet
        pnl = 0.0
        if a == 0:
            pnl = 0.0
        else:
            # fixed stake fraction * current bankroll, clipped
            stake_frac = min(float(cfg.stake_frac), float(cfg.max_stake_frac))
            stake = bankroll_before * stake_frac

            # stop episode early if bankroll or stake becomes too small
            if float(bankroll_before) <= float(cfg.min_bankroll) or float(stake) < float(cfg.min_stake):
                # trim arrays and finish episode
                obs_mat = obs_mat[: t + 1]
                probs_mat = probs_mat[: t + 1]
                actions = actions[: t + 1]
                rewards = rewards[: t + 1]
                break

            pick = a - 1  # 0=home,1=draw,2=away
            true = int(outcome[t])
            odds = float([ho[t], do[t], ao[t]][pick])

            n_bets += 1
            if pick == true:
                pnl = stake * (odds - 1.0)
                n_wins += 1
            else:
                pnl = -stake

        bankroll = max(0.0, bankroll_before + pnl)
        total_pnl += float(pnl)

        if cfg.reward_mode == "profit":
            r = float(pnl)
        else:
            r = float(math.log(max(bankroll, eps)) - math.log(max(bankroll_before, eps)))

        # small per-bet penalty to discourage betting on marginal / negative EV
        if a != 0:
            r -= float(cfg.bet_penalty)

        obs_mat[t] = x_in
        probs_mat[t] = probs
        actions[t] = a
        rewards[t] = r

        if bankroll <= 0.0:
            # "ruin" => end episode early; trim arrays
            obs_mat = obs_mat[: t + 1]
            probs_mat = probs_mat[: t + 1]
            actions = actions[: t + 1]
            rewards = rewards[: t + 1]
            break

    stats = {
        "final_bankroll": float(bankroll),
        "total_pnl": float(total_pnl),
        "n_steps": float(len(rewards)),
        "n_bets": float(n_bets),
        "win_rate": float(n_wins / max(n_bets, 1)),
        "sum_reward": float(np.sum(rewards)),
    }
    return obs_mat, probs_mat, actions, {"rewards": rewards, **stats}


# ----------------------------
# Training loop (REINFORCE)
# ----------------------------

def train(
    train_arrays: Dict[str, np.ndarray],
    test_arrays: Optional[Dict[str, np.ndarray]],
    *,
    cfg: TrainConfig,
    out_path: Optional[str],
) -> None:
    rng = np.random.default_rng(int(cfg.seed))

    n_actions = 4
    obs_dim = 14  # must match observation vector in run_episode() (added low1 feature)

    # Simple linear policy: pi(a|s) = softmax(W s + b)
    W = rng.normal(loc=0.0, scale=0.01, size=(n_actions, obs_dim))
    b = np.zeros(n_actions, dtype=float)

    obs_norm = RunningNorm(obs_dim) if cfg.use_obs_norm else None

    baseline_ema = 0.0
    beta = 0.9  # baseline smoothing

    for epoch in range(1, int(cfg.epochs) + 1):
        obs_mat, probs_mat, actions, info = run_episode(
            train_arrays,
            W=W,
            b=b,
            rng=rng,
            cfg=cfg,
            greedy=False,
            obs_norm=obs_norm,
        )
        rewards = info["rewards"]
        returns = _discounted_cumsum(rewards, gamma=float(cfg.gamma))

        # simple baseline (EMA of episode mean return)
        ep_mean_return = float(np.mean(returns)) if len(returns) else 0.0
        baseline_ema = beta * baseline_ema + (1.0 - beta) * ep_mean_return

        adv = returns - baseline_ema
        # extra variance reduction: center + normalize advantages to stabilize gradients
        adv = adv - float(np.mean(adv))
        adv_std = float(np.std(adv))
        if adv_std > 0:
            adv = adv / max(adv_std, 1e-8)

        grad_W = np.zeros_like(W)
        grad_b = np.zeros_like(b)

        for t in range(len(rewards)):
            p = probs_mat[t]  # shape [4]
            a = int(actions[t])
            x = obs_mat[t]    # shape [obs_dim]
            # grad log softmax: (1[a] - p)
            g = -p.copy()
            g[a] += 1.0
            grad_W += float(adv[t]) * np.outer(g, x)
            grad_b += float(adv[t]) * g

        # normalize + clip
        T = max(len(rewards), 1)
        grad_W /= float(T)
        grad_b /= float(T)

        gnorm = float(np.sqrt(np.sum(grad_W ** 2) + np.sum(grad_b ** 2)))
        if gnorm > float(cfg.grad_clip_norm) and gnorm > 0:
            scale = float(cfg.grad_clip_norm) / gnorm
            grad_W *= scale
            grad_b *= scale

        # gradient ascent
        W += float(cfg.lr) * grad_W
        b += float(cfg.lr) * grad_b

        if (epoch == 1) or (epoch % max(int(cfg.eval_every), 1) == 0) or (epoch == int(cfg.epochs)):
            msg = (
                f"[epoch {epoch:03d}] "
                f"train_final_bankroll={info['final_bankroll']:.2f}  "
                f"train_pnl={info['total_pnl']:.2f}  "
                f"train_bets={int(info['n_bets'])}  "
                f"train_winrate={info['win_rate']:.2%}  "
                f"sum_reward={info['sum_reward']:.4f}"
            )
            print(msg)

            if test_arrays is not None:
                _, _, _, test_info = run_episode(
                    test_arrays,
                    W=W,
                    b=b,
                    rng=rng,
                    cfg=cfg,
                    greedy=True,  # greedy eval
                    obs_norm=obs_norm,
                )
                print(
                    f"           test_final_bankroll={test_info['final_bankroll']:.2f}  "
                    f"test_pnl={test_info['total_pnl']:.2f}  "
                    f"test_bets={int(test_info['n_bets'])}  "
                    f"test_winrate={test_info['win_rate']:.2%}  "
                    f"sum_reward={test_info['sum_reward']:.4f}"
                )

    if out_path:
        payload = {
            "W": W,
            "b": b,
            "obs_dim": int(obs_dim),
            "action_names": ["skip", "bet_home", "bet_draw", "bet_away"],
            "note": "Linear softmax policy trained with REINFORCE",
            "train_cfg": cfg.__dict__,
        }

        # persist obs_norm stats when available so evaluation is deterministic
        if obs_norm is not None:
            try:
                payload["obs_norm"] = {
                    "mean": obs_norm.mean.tolist(),
                    "std": obs_norm.std().tolist(),
                    "n": int(getattr(obs_norm, "n", 0)),
                }
            except Exception:
                # best-effort: don't fail saving the policy over normalization metadata
                pass

        joblib.dump(payload, out_path)
        print(f"\nSaved policy to: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Minimal RL policy training wrapper (REINFORCE)")

    p.add_argument("--artifact", type=str, default=DEFAULT_ARTIFACT_ABS_PATH)
    p.add_argument("--matches", type=str, default=DEFAULT_MATCHES_ABS_PATH)
    p.add_argument("--stadiums", type=str, default=DEFAULT_STADIUMS_ABS_PATH)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=7)

    p.add_argument("--initial-bankroll", type=float, default=1000.0)
    p.add_argument("--stake-frac", type=float, default=0.01)
    p.add_argument("--max-stake-frac", type=float, default=0.05)
    p.add_argument("--reward-mode", choices=["log_growth", "profit"], default="log_growth")

    p.add_argument("--no-obs-norm", action="store_true")
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--eval-every", type=int, default=5)

    p.add_argument("--min-bankroll", type=float, default=10.0, help="Stop episode when bankroll falls below this value.")
    p.add_argument("--min-stake", type=float, default=1.0, help="Stop episode when computed stake (bankroll*stake_frac) falls below this value.")
    p.add_argument("--ev-threshold", type=float, default=0.01, help="Minimum model EV per unit required to allow betting actions during training.")
    p.add_argument("--bet-penalty", type=float, default=0.001, help="Per-bet penalty subtracted from reward to discourage over-betting.")
    p.add_argument("--low-score-penalty", type=float, default=0.0, help="Per-unit low-score probability penalty added to reward.")

    p.add_argument("--save", type=str, default="models/rl_policy.joblib")
    p.add_argument("--no-test-eval", action="store_true")

    args = p.parse_args()

    cfg = TrainConfig(
        epochs=int(args.epochs),
        lr=float(args.lr),
        gamma=float(args.gamma),
        seed=int(args.seed),
        initial_bankroll=float(args.initial_bankroll),
        stake_frac=float(args.stake_frac),
        max_stake_frac=float(args.max_stake_frac),
        min_bankroll=float(args.min_bankroll),
        min_stake=float(args.min_stake),
        ev_threshold=float(args.ev_threshold),
        bet_penalty=float(args.bet_penalty),
        low_score_penalty=float(args.low_score_penalty),
        reward_mode=str(args.reward_mode),
        use_obs_norm=(not bool(args.no_obs_norm)),
        grad_clip_norm=float(args.grad_clip),
        eval_every=int(args.eval_every),
    )

    artifact_path = Path(args.artifact)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")

    train_arrays = build_precomputed_arrays(
        artifact_path=str(args.artifact),
        matches_path=str(args.matches),
        stadiums_path=str(args.stadiums) if args.stadiums else None,
        split="train",
    )

    test_arrays = None
    if not bool(args.no_test_eval):
        test_arrays = build_precomputed_arrays(
            artifact_path=str(args.artifact),
            matches_path=str(args.matches),
            stadiums_path=str(args.stadiums) if args.stadiums else None,
            split="test",
        )

    out_path = str(args.save) if args.save else None
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    train(train_arrays, test_arrays, cfg=cfg, out_path=out_path)


if __name__ == "__main__":
    main()
