"""
train.py

Train two Poisson models:
  - model_home: predicts expected home goals
  - model_away: predicts expected away goals

Usage:
  python train.py --matches data/spi_matches.csv --stadiums data/stadium_coordinates_completed_full.csv --out models

This script:
  1) builds leakage-safe rolling features for each historical match
  2) performs a clean time-based split
  3) evaluates on the test period
  4) writes a joblib artifact for use in Streamlit app.py

"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import DEFAULT_MATCHES_REL_PATH, DEFAULT_MODELS_DIR, DEFAULT_STADIUMS_REL_PATH, DEFAULT_ARTIFACT_FILENAME
from features import (
    FeatureConfig,
    build_training_frame,
    load_matches_csv,
    load_stadiums_csv,
    model_categorical_columns,
    model_numeric_columns,
)
from metrics import (
    fit_dixon_coles_rho,
    expected_goals_regression_metrics,
    neg_log_likelihood,
    neg_log_likelihood_dixon_coles,
    per_match_neg_log_likelihood_dixon_coles,
)
from predict import scoreline_probability_matrix, calibrate_scoreline_matrix


def _installed_version(dist_name: str) -> str | None:
    try:
        return package_version(dist_name)
    except PackageNotFoundError:
        return None


def build_time_decay_weights(
    dates: pd.Series,
    ref_date: pd.Timestamp,
    half_life_days: float,
    min_weight: float = 1e-6,
) -> np.ndarray:
    """
    Exponential time-decay weights:
      w = 0.5 ** (delta_days / half_life_days)
    with delta_days measured from ref_date backwards.

    If half_life_days <= 0, returns all ones.
    """
    if float(half_life_days) <= 0:
        return np.ones(len(dates), dtype=float)

    delta_days = (pd.Timestamp(ref_date) - pd.to_datetime(dates)).dt.days.astype(float).to_numpy()
    delta_days = np.clip(delta_days, 0.0, None)
    weights = np.power(0.5, delta_days / float(half_life_days))
    return np.clip(weights, float(min_weight), None)


def build_pipeline(cat_cols: Tuple[str, ...], num_cols: Tuple[str, ...],
                   alpha: float = 1e-4, max_iter: int = 500) -> Pipeline:
    """
    Pipeline:
      - impute + one-hot categorical vars
      - impute numeric vars
      - PoissonRegressor
    """
    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("cat", categorical, list(cat_cols)),
            ("num", numeric, list(num_cols)),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    model = PoissonRegressor(alpha=alpha, max_iter=max_iter)
    return Pipeline(steps=[("preprocess", pre), ("model", model)])


def tune_decay_half_life(
    *,
    train_df: pd.DataFrame,
    cat_cols: Tuple[str, ...],
    num_cols: Tuple[str, ...],
    candidates: List[float],
    val_days: int,
    alpha: float,
    max_iter: int,
    tune_metric: str,
    fit_dc: bool,
    dc_rho_min: float,
    dc_rho_max: float,
    dc_coarse_steps: int,
    dc_fine_steps: int,
    dc_optimize_oot: bool,
    dc_significance_z: float,
) -> Dict[str, object]:
    """
    Tune half-life on a validation window from the END of the training period.
    This avoids tuning on the held-out test set.
    """
    train_df = train_df.sort_values("date").copy()
    max_train_date = pd.Timestamp(train_df["date"].max())
    val_cutoff = max_train_date - pd.Timedelta(days=int(val_days))

    fit_df = train_df[train_df["date"] < val_cutoff].copy()
    val_df = train_df[train_df["date"] >= val_cutoff].copy()

    if len(fit_df) < 5000 or len(val_df) < 1000:
        return {
            "best_half_life_days": float(candidates[0]),
            "val_cutoff_date": str(val_cutoff.date()),
            "fit_rows": int(len(fit_df)),
            "val_rows": int(len(val_df)),
            "results": [],
            "warning": "Not enough data for tuning split; skipped tuning.",
        }

    if tune_metric == "dc_nll" and not fit_dc:
        raise ValueError("tune_metric='dc_nll' requires --fit-dc.")

    X_fit = fit_df[list(cat_cols) + list(num_cols)]
    y_home_fit = fit_df["home_goals"].astype(int).to_numpy()
    y_away_fit = fit_df["away_goals"].astype(int).to_numpy()

    X_val = val_df[list(cat_cols) + list(num_cols)]
    y_home_val = val_df["home_goals"].astype(int).to_numpy()
    y_away_val = val_df["away_goals"].astype(int).to_numpy()

    ref_date_fit = pd.Timestamp(fit_df["date"].max())
    min_weight = 1e-6

    results = []
    best = {"half_life_days": None, "metric": float("inf"), "rho": None, "rho_source": None}
    cand = sorted({float(c) for c in candidates})

    for hl in cand:
        w_fit = build_time_decay_weights(
            dates=fit_df["date"],
            ref_date=ref_date_fit,
            half_life_days=float(hl),
            min_weight=min_weight,
        )

        model_home = build_pipeline(cat_cols, num_cols, alpha=alpha, max_iter=max_iter)
        model_away = build_pipeline(cat_cols, num_cols, alpha=alpha, max_iter=max_iter)
        model_home.fit(X_fit, y_home_fit, model__sample_weight=w_fit)
        model_away.fit(X_fit, y_away_fit, model__sample_weight=w_fit)

        lam_home_val = np.clip(model_home.predict(X_val), 1e-6, None)
        lam_away_val = np.clip(model_away.predict(X_val), 1e-6, None)

        if tune_metric == "ind_nll":
            metric = neg_log_likelihood(y_home_val, y_away_val, lam_home_val, lam_away_val)
            rho_used = None
            rho_source = None
        else:
            if dc_optimize_oot:
                rho_raw, _ = fit_dixon_coles_rho(
                    y_home=y_home_val,
                    y_away=y_away_val,
                    lam_home=lam_home_val,
                    lam_away=lam_away_val,
                    rho_min=dc_rho_min,
                    rho_max=dc_rho_max,
                    coarse_steps=dc_coarse_steps,
                    fine_steps=dc_fine_steps,
                    sample_weight=None,
                )
                rho_used, metric, rho_test = choose_rho_with_significance(
                    y_home=y_home_val,
                    y_away=y_away_val,
                    lam_home=lam_home_val,
                    lam_away=lam_away_val,
                    rho_candidate=float(rho_raw),
                    z_critical=float(dc_significance_z),
                )
                rho_source = "oot_validation_significance"
            else:
                lam_home_fit = np.clip(model_home.predict(X_fit), 1e-6, None)
                lam_away_fit = np.clip(model_away.predict(X_fit), 1e-6, None)
                rho, _ = fit_dixon_coles_rho(
                    y_home=y_home_fit,
                    y_away=y_away_fit,
                    lam_home=lam_home_fit,
                    lam_away=lam_away_fit,
                    rho_min=dc_rho_min,
                    rho_max=dc_rho_max,
                    coarse_steps=dc_coarse_steps,
                    fine_steps=dc_fine_steps,
                    sample_weight=w_fit,
                )
                metric = neg_log_likelihood_dixon_coles(
                    y_home_val,
                    y_away_val,
                    lam_home_val,
                    lam_away_val,
                    rho=float(rho),
                )
                rho_used = float(rho)
                rho_source = "fit_split"
                rho_test = None

        row = {
            "half_life_days": float(hl),
            "val_metric": float(metric),
            "metric_name": tune_metric,
            "rho_used": rho_used,
            "rho_source": rho_source,
            "rho_test": rho_test,
        }
        results.append(row)

        if float(metric) < float(best["metric"]):
            best = {
                "half_life_days": float(hl),
                "metric": float(metric),
                "rho": rho_used,
                "rho_source": rho_source,
            }

    results_sorted = sorted(results, key=lambda r: r["val_metric"])

    return {
        "best_half_life_days": float(best["half_life_days"]),
        "best_val_metric": float(best["metric"]),
        "best_rho": best.get("rho"),
        "best_rho_source": best.get("rho_source"),
        "metric_name": tune_metric,
        "val_cutoff_date": str(val_cutoff.date()),
        "fit_rows": int(len(fit_df)),
        "val_rows": int(len(val_df)),
        "results": results_sorted,
    }


def low_score_calibration_summary(
    y_home: np.ndarray,
    y_away: np.ndarray,
    lam_home: np.ndarray,
    lam_away: np.ndarray,
    rho: float | None = None,
) -> dict[str, dict[str, float]]:
    """
    Empirical vs predicted frequencies for low scores on a dataset.
    Scores: 0-0, 1-0, 0-1, 1-1.
    """
    lh = np.clip(lam_home.astype(float), 1e-12, None)
    la = np.clip(lam_away.astype(float), 1e-12, None)

    ph0 = np.exp(-lh)
    ph1 = ph0 * lh
    pa0 = np.exp(-la)
    pa1 = pa0 * la

    p00 = ph0 * pa0
    p10 = ph1 * pa0
    p01 = ph0 * pa1
    p11 = ph1 * pa1

    if rho is not None:
        r = float(rho)
        tau00 = np.maximum(1e-12, 1.0 - (lh * la * r))
        tau01 = np.maximum(1e-12, 1.0 + (lh * r))
        tau10 = np.maximum(1e-12, 1.0 + (la * r))
        tau11 = np.maximum(1e-12, 1.0 - r)

        z = 1.0 + p00 * (tau00 - 1.0) + p01 * (tau01 - 1.0) + p10 * (tau10 - 1.0) + p11 * (tau11 - 1.0)
        z = np.maximum(1e-12, z)

        p00 = (p00 * tau00) / z
        p01 = (p01 * tau01) / z
        p10 = (p10 * tau10) / z
        p11 = (p11 * tau11) / z

    obs00 = (y_home == 0) & (y_away == 0)
    obs10 = (y_home == 1) & (y_away == 0)
    obs01 = (y_home == 0) & (y_away == 1)
    obs11 = (y_home == 1) & (y_away == 1)

    out = {
        "0-0": {"empirical": float(np.mean(obs00)), "predicted": float(np.mean(p00))},
        "1-0": {"empirical": float(np.mean(obs10)), "predicted": float(np.mean(p10))},
        "0-1": {"empirical": float(np.mean(obs01)), "predicted": float(np.mean(p01))},
        "1-1": {"empirical": float(np.mean(obs11)), "predicted": float(np.mean(p11))},
    }

    for score in out:
        out[score]["delta_pp"] = 100.0 * (out[score]["predicted"] - out[score]["empirical"])

    return out


def evaluation_accuracy_summary(
    y_home: np.ndarray,
    y_away: np.ndarray,
    lam_home: np.ndarray,
    lam_away: np.ndarray,
    max_goals: int,
    rho: float | None = None,
    calibration_cfg: Optional[Dict[str, object]] = None,
    divs: Optional[np.ndarray] = None,
) -> dict[str, float]:
    exact_correct = 0
    wdl_correct = 0
    n = len(y_home)

    for i in range(n):
        mat = scoreline_probability_matrix(
            float(lam_home[i]),
            float(lam_away[i]),
            max_goals=int(max_goals),
            include_tail_bucket=True,
            rho=float(rho) if rho is not None else None,
        )
        div_i = str(divs[i]) if divs is not None else None
        mat = calibrate_scoreline_matrix(mat, calibration_cfg, div=div_i)
        arr = mat.to_numpy()
        idx = int(np.argmax(arr))
        r, c = divmod(idx, arr.shape[1])
        pred_h = str(mat.index[r])
        pred_a = str(mat.columns[c])

        if (not pred_h.endswith("+")) and (not pred_a.endswith("+")):
            exact_correct += int((int(pred_h) == int(y_home[i])) and (int(pred_a) == int(y_away[i])))

        home_win = float(np.tril(arr, k=-1).sum())
        draw = float(np.trace(arr))
        away_win = float(np.triu(arr, k=1).sum())
        pred_outcome = max(
            {"home_win": home_win, "draw": draw, "away_win": away_win},
            key=lambda k: {"home_win": home_win, "draw": draw, "away_win": away_win}[k],
        )
        if int(y_home[i]) > int(y_away[i]):
            true_outcome = "home_win"
        elif int(y_home[i]) == int(y_away[i]):
            true_outcome = "draw"
        else:
            true_outcome = "away_win"
        wdl_correct += int(pred_outcome == true_outcome)

    denom = max(n, 1)
    return {
        "exact_score_top1_acc": float(exact_correct / denom),
        "wdl_acc": float(wdl_correct / denom),
    }


def top_score_frequency_distribution(
    lam_home: np.ndarray,
    lam_away: np.ndarray,
    max_goals: int,
    rho: float | None = None,
    top_n: int = 8,
    calibration_cfg: Optional[Dict[str, object]] = None,
    divs: Optional[np.ndarray] = None,
) -> dict[str, object]:
    n = len(lam_home)
    counts: Dict[str, int] = {}

    for i in range(n):
        mat = scoreline_probability_matrix(
            float(lam_home[i]),
            float(lam_away[i]),
            max_goals=int(max_goals),
            include_tail_bucket=True,
            rho=float(rho) if rho is not None else None,
        )
        div_i = str(divs[i]) if divs is not None else None
        mat = calibrate_scoreline_matrix(mat, calibration_cfg, div=div_i)
        arr = mat.to_numpy()
        idx = int(np.argmax(arr))
        r, c = divmod(idx, arr.shape[1])
        key = f"{mat.index[r]}-{mat.columns[c]}"
        counts[key] = counts.get(key, 0) + 1

    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    out = []
    denom = max(n, 1)
    for score, cnt in ranked[: int(top_n)]:
        out.append({
            "score": score,
            "count": int(cnt),
            "share": float(cnt / denom),
        })

    return {
        "n_matches": int(n),
        "top_n": int(top_n),
        "rows": out,
    }


def top_score_mode_share(
    lam_home: np.ndarray,
    lam_away: np.ndarray,
    max_goals: int,
    rho: float | None = None,
    calibration_cfg: Optional[Dict[str, object]] = None,
    divs: Optional[np.ndarray] = None,
) -> float:
    diag = top_score_frequency_distribution(
        lam_home,
        lam_away,
        max_goals=max_goals,
        rho=rho,
        top_n=1,
        calibration_cfg=calibration_cfg,
        divs=divs,
    )
    rows = diag.get("rows", [])
    if not rows:
        return 0.0
    return float(rows[0].get("share", 0.0))


def shrink_rho_for_concentration(
    lam_home: np.ndarray,
    lam_away: np.ndarray,
    rho: float,
    max_goals: int,
    max_top_share: float,
    steps: int = 18,
    calibration_cfg: Optional[Dict[str, object]] = None,
    divs: Optional[np.ndarray] = None,
) -> tuple[float, float]:
    """
    If mode share exceeds threshold, shrink rho toward 0 with binary search.
    Returns (rho_adjusted, achieved_mode_share).
    """
    target = float(max_top_share)
    current_share = top_score_mode_share(
        lam_home,
        lam_away,
        max_goals=max_goals,
        rho=float(rho),
        calibration_cfg=calibration_cfg,
        divs=divs,
    )
    if current_share <= target:
        return float(rho), float(current_share)

    lo = 0.0
    hi = 1.0
    best_scale = 0.0

    for _ in range(int(steps)):
        mid = 0.5 * (lo + hi)
        rho_mid = float(rho) * mid
        share_mid = top_score_mode_share(
            lam_home,
            lam_away,
            max_goals=max_goals,
            rho=rho_mid,
            calibration_cfg=calibration_cfg,
            divs=divs,
        )
        if share_mid <= target:
            best_scale = mid
            lo = mid
        else:
            hi = mid

    rho_adj = float(rho) * best_scale
    share_adj = top_score_mode_share(
        lam_home,
        lam_away,
        max_goals=max_goals,
        rho=rho_adj,
        calibration_cfg=calibration_cfg,
        divs=divs,
    )
    return rho_adj, share_adj


def _score_to_label(v: int, max_goals: int) -> str:
    return str(int(v)) if int(v) <= int(max_goals) else f"{int(max_goals)}+"


def fit_temperature_calibration(
    y_home: np.ndarray,
    y_away: np.ndarray,
    lam_home: np.ndarray,
    lam_away: np.ndarray,
    max_goals: int,
    rho: float | None,
    temp_min: float = 0.6,
    temp_max: float = 2.0,
    steps: int = 57,
) -> dict[str, object]:
    labels = [str(i) for i in range(int(max_goals) + 1)] + [f"{int(max_goals)}+"]
    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    mats = []
    true_idx = []
    for i in range(len(y_home)):
        mat = scoreline_probability_matrix(
            float(lam_home[i]),
            float(lam_away[i]),
            max_goals=int(max_goals),
            include_tail_bucket=True,
            rho=float(rho) if rho is not None else None,
        )
        arr = np.clip(mat.to_numpy(dtype=float).reshape(-1), 1e-12, None)
        arr = arr / max(float(arr.sum()), 1e-12)
        mats.append(arr)

        lab_h = _score_to_label(int(y_home[i]), int(max_goals))
        lab_a = _score_to_label(int(y_away[i]), int(max_goals))
        r = label_to_idx[lab_h]
        c = label_to_idx[lab_a]
        true_idx.append(r * len(labels) + c)

    P = np.vstack(mats)
    y_idx = np.asarray(true_idx, dtype=int)

    def mean_nll_for_temp(t: float) -> float:
        inv_t = 1.0 / float(max(t, 1e-6))
        S = np.power(P, inv_t)
        S = S / np.clip(S.sum(axis=1, keepdims=True), 1e-12, None)
        p_true = np.clip(S[np.arange(len(S)), y_idx], 1e-12, None)
        return float(-np.mean(np.log(p_true)))

    grid = np.linspace(float(temp_min), float(temp_max), int(steps))
    best_t = 1.0
    best_nll = float("inf")
    for t in grid:
        nll_t = mean_nll_for_temp(float(t))
        if nll_t < best_nll:
            best_nll = nll_t
            best_t = float(t)

    nll_uncal = mean_nll_for_temp(1.0)

    return {
        "enabled": True,
        "method": "temperature",
        "temperature": float(best_t),
        "nll_uncal": float(nll_uncal),
        "nll_cal": float(best_nll),
        "nll_improvement": float(nll_uncal - best_nll),
        "n_rows": int(len(y_home)),
    }


def fit_low_score_mixture_calibration(
    y_home: np.ndarray,
    y_away: np.ndarray,
    lam_home: np.ndarray,
    lam_away: np.ndarray,
    max_goals: int,
    rho: float | None,
    calibration_cfg: Optional[Dict[str, object]],
    events: tuple[str, ...] = ("0-0", "1-0", "0-1", "1-1"),
    alpha_min: float = 0.0,
    alpha_max: float = 0.5,
    steps: int = 41,
) -> dict[str, object]:
    labels = [str(i) for i in range(int(max_goals) + 1)] + [f"{int(max_goals)}+"]
    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    mats = []
    true_idx = []
    for i in range(len(y_home)):
        mat = scoreline_probability_matrix(
            float(lam_home[i]),
            float(lam_away[i]),
            max_goals=int(max_goals),
            include_tail_bucket=True,
            rho=float(rho) if rho is not None else None,
        )
        div_i = None
        mat = calibrate_scoreline_matrix(mat, calibration_cfg, div=div_i)
        arr = np.clip(mat.to_numpy(dtype=float), 1e-12, None)
        arr = arr / max(float(arr.sum()), 1e-12)
        mats.append(arr)

        lab_h = _score_to_label(int(y_home[i]), int(max_goals))
        lab_a = _score_to_label(int(y_away[i]), int(max_goals))
        r = label_to_idx[lab_h]
        c = label_to_idx[lab_a]
        true_idx.append((r, c))

    emp = {}
    n = max(len(y_home), 1)
    for ev in events:
        eh, ea = ev.split("-")
        emp[ev] = float(np.mean((y_home == int(eh)) & (y_away == int(ea)))) if len(y_home) else 0.0

    emp_sum = float(sum(emp.values()))
    if emp_sum <= 0:
        return {
            "enabled": False,
            "method": "low_score_mixture",
            "alpha": 0.0,
            "target_probs": emp,
            "n_rows": int(len(y_home)),
            "reason": "No low-score empirical mass in calibration split.",
        }
    if emp_sum > 1.0:
        emp = {k: v / emp_sum for k, v in emp.items()}

    event_pos = []
    for ev, p in emp.items():
        eh, ea = ev.split("-")
        if eh in label_to_idx and ea in label_to_idx:
            event_pos.append((label_to_idx[eh], label_to_idx[ea], float(p)))

    if not event_pos:
        return {
            "enabled": False,
            "method": "low_score_mixture",
            "alpha": 0.0,
            "target_probs": emp,
            "n_rows": int(len(y_home)),
            "reason": "No valid low-score event positions for current max_goals.",
        }

    def _mix_arr(arr: np.ndarray, alpha: float) -> np.ndarray:
        p = np.asarray(arr, dtype=float)
        p = p / max(float(p.sum()), 1e-12)
        q = np.zeros_like(p, dtype=float)
        mask = np.zeros_like(p, dtype=bool)
        for r, c, prob in event_pos:
            q[r, c] = prob
            mask[r, c] = True

        rem = max(0.0, 1.0 - float(sum(prob for _, _, prob in event_pos)))
        non = p[~mask]
        non_sum = float(non.sum())
        if non_sum > 0:
            q[~mask] = non * (rem / non_sum)
        elif (~mask).any():
            q[~mask] = rem / float((~mask).sum())

        m = (1.0 - alpha) * p + alpha * q
        return m / max(float(m.sum()), 1e-12)

    def _nll(alpha: float) -> float:
        loss = 0.0
        for arr, (r, c) in zip(mats, true_idx):
            m = _mix_arr(arr, alpha)
            loss += -math.log(max(float(m[r, c]), 1e-12))
        return float(loss / max(len(mats), 1))

    base_nll = _nll(0.0)
    best_alpha = 0.0
    best_nll = base_nll
    for alpha in np.linspace(float(alpha_min), float(alpha_max), int(steps)):
        nll = _nll(float(alpha))
        if nll < best_nll:
            best_nll = nll
            best_alpha = float(alpha)

    return {
        "enabled": bool(best_alpha > 0.0),
        "method": "low_score_mixture",
        "alpha": float(best_alpha),
        "alpha_min": float(alpha_min),
        "alpha_max": float(alpha_max),
        "steps": int(steps),
        "events": list(events),
        "target_probs": {k: float(v) for k, v in emp.items()},
        "nll_base": float(base_nll),
        "nll_mix": float(best_nll),
        "nll_improvement": float(base_nll - best_nll),
        "n_rows": int(n),
    }


def top1_reliability_report(
    y_home: np.ndarray,
    y_away: np.ndarray,
    lam_home: np.ndarray,
    lam_away: np.ndarray,
    max_goals: int,
    rho: float | None,
    calibration_cfg: Optional[Dict[str, object]],
    n_bins: int = 10,
    divs: Optional[np.ndarray] = None,
) -> dict[str, object]:
    conf = []
    corr = []

    for i in range(len(y_home)):
        mat = scoreline_probability_matrix(
            float(lam_home[i]),
            float(lam_away[i]),
            max_goals=int(max_goals),
            include_tail_bucket=True,
            rho=float(rho) if rho is not None else None,
        )
        div_i = str(divs[i]) if divs is not None else None
        mat = calibrate_scoreline_matrix(mat, calibration_cfg, div=div_i)
        arr = mat.to_numpy(dtype=float)
        idx = int(np.argmax(arr))
        r, c = divmod(idx, arr.shape[1])
        pred_h = str(mat.index[r])
        pred_a = str(mat.columns[c])
        true_h = _score_to_label(int(y_home[i]), int(max_goals))
        true_a = _score_to_label(int(y_away[i]), int(max_goals))

        conf.append(float(arr[r, c]))
        corr.append(1.0 if (pred_h == true_h and pred_a == true_a) else 0.0)

    conf = np.asarray(conf, dtype=float)
    corr = np.asarray(corr, dtype=float)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    rows = []
    ece = 0.0
    for b in range(int(n_bins)):
        lo = edges[b]
        hi = edges[b + 1]
        mask = (conf >= lo) & (conf < hi if b < n_bins - 1 else conf <= hi)
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        conf_b = float(conf[mask].mean())
        acc_b = float(corr[mask].mean())
        gap = abs(acc_b - conf_b)
        ece += (n_b / max(len(conf), 1)) * gap
        rows.append(
            {
                "bin": f"[{lo:.1f},{hi:.1f}{')' if b < n_bins - 1 else ']'}",
                "n": n_b,
                "mean_conf": conf_b,
                "empirical_acc": acc_b,
                "gap": float(acc_b - conf_b),
            }
        )

    return {
        "n": int(len(conf)),
        "ece": float(ece),
        "rows": rows,
    }


def event_level_reliability_report(
    y_home: np.ndarray,
    y_away: np.ndarray,
    lam_home: np.ndarray,
    lam_away: np.ndarray,
    max_goals: int,
    rho: float | None,
    calibration_cfg: Optional[Dict[str, object]],
    events: tuple[str, ...] = ("1-1", "1-0", "0-1", "0-0"),
    n_bins: int = 10,
    divs: Optional[np.ndarray] = None,
) -> dict[str, object]:
    labels = [str(i) for i in range(int(max_goals) + 1)] + [f"{int(max_goals)}+"]
    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    event_probs: Dict[str, list[float]] = {e: [] for e in events}
    event_obs: Dict[str, list[float]] = {e: [] for e in events}

    for i in range(len(y_home)):
        mat = scoreline_probability_matrix(
            float(lam_home[i]),
            float(lam_away[i]),
            max_goals=int(max_goals),
            include_tail_bucket=True,
            rho=float(rho) if rho is not None else None,
        )
        div_i = str(divs[i]) if divs is not None else None
        mat = calibrate_scoreline_matrix(mat, calibration_cfg, div=div_i)
        arr = mat.to_numpy(dtype=float)

        true_label = f"{_score_to_label(int(y_home[i]), int(max_goals))}-{_score_to_label(int(y_away[i]), int(max_goals))}"
        for event in events:
            eh, ea = event.split("-")
            if eh not in label_to_idx or ea not in label_to_idx:
                continue
            r = label_to_idx[eh]
            c = label_to_idx[ea]
            event_probs[event].append(float(arr[r, c]))
            event_obs[event].append(1.0 if true_label == event else 0.0)

    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    result_events: Dict[str, dict[str, object]] = {}
    for event in events:
        probs = np.asarray(event_probs[event], dtype=float)
        obs = np.asarray(event_obs[event], dtype=float)
        if len(probs) == 0:
            continue

        rows = []
        ece = 0.0
        for b in range(int(n_bins)):
            lo = edges[b]
            hi = edges[b + 1]
            mask = (probs >= lo) & (probs < hi if b < n_bins - 1 else probs <= hi)
            n_b = int(mask.sum())
            if n_b == 0:
                continue
            p_b = float(probs[mask].mean())
            o_b = float(obs[mask].mean())
            ece += (n_b / max(len(probs), 1)) * abs(o_b - p_b)
            rows.append(
                {
                    "bin": f"[{lo:.1f},{hi:.1f}{')' if b < n_bins - 1 else ']'}",
                    "n": n_b,
                    "mean_pred": p_b,
                    "empirical_rate": o_b,
                    "gap": float(o_b - p_b),
                }
            )

        result_events[event] = {
            "n": int(len(probs)),
            "prevalence": float(obs.mean()),
            "mean_pred": float(probs.mean()),
            "brier": float(np.mean((probs - obs) ** 2)),
            "ece": float(ece),
            "rows": rows,
        }

    return {
        "n": int(len(y_home)),
        "events": result_events,
    }


def benchmark_panel(
    y_home_train: np.ndarray,
    y_away_train: np.ndarray,
    y_home_test: np.ndarray,
    y_away_test: np.ndarray,
    model_top1_acc: float,
    model_nll_ind: float,
) -> dict[str, float]:
    train_pairs = list(zip(y_home_train.astype(int), y_away_train.astype(int)))
    test_pairs = list(zip(y_home_test.astype(int), y_away_test.astype(int)))

    train_counts: Dict[tuple[int, int], int] = {}
    for pair in train_pairs:
        train_counts[pair] = train_counts.get(pair, 0) + 1
    total_train = max(len(train_pairs), 1)
    probs = {k: v / total_train for k, v in train_counts.items()}

    most_common = max(train_counts.items(), key=lambda kv: kv[1])[0] if train_counts else (1, 1)
    always_mode_acc = float(np.mean([(p == most_common) for p in test_pairs]))
    always_11_acc = float(np.mean((y_home_test == 1) & (y_away_test == 1)))

    max_goal_obs = int(max(np.max(y_home_train), np.max(y_away_train), 1))
    uniform_exact_acc = 1.0 / float((max_goal_obs + 1) ** 2)

    empirical_random_acc = float(sum((v / total_train) * probs.get(k, 0.0) for k, v in train_counts.items()))
    eps = 1e-12
    empirical_nll = float(-np.mean([np.log(max(probs.get(p, eps), eps)) for p in test_pairs]))

    return {
        "model_top1_acc": float(model_top1_acc),
        "model_nll_ind": float(model_nll_ind),
        "always_train_mode_acc": always_mode_acc,
        "always_1_1_acc": always_11_acc,
        "uniform_random_acc": uniform_exact_acc,
        "empirical_random_acc": empirical_random_acc,
        "empirical_nll": empirical_nll,
    }


def diebold_mariano_test(loss_model: np.ndarray, loss_baseline: np.ndarray) -> dict[str, float | int]:
    d = np.asarray(loss_model, dtype=float) - np.asarray(loss_baseline, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 3:
        return {"n": int(n), "mean_diff": float("nan"), "se": float("nan"), "z": float("nan"), "p_value": float("nan")}

    mean_diff = float(np.mean(d))
    se = float(np.std(d, ddof=1) / np.sqrt(n))
    z = float(mean_diff / max(se, 1e-12))
    p = float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0)))))
    return {"n": int(n), "mean_diff": mean_diff, "se": se, "z": z, "p_value": p}


def lambda_interval_diagnostics(y: np.ndarray, lam: np.ndarray, z: float = 1.96) -> dict[str, float]:
    lam = np.clip(np.asarray(lam, dtype=float), 1e-6, None)
    y = np.asarray(y, dtype=float)
    sd = np.sqrt(lam)
    lo = np.maximum(0.0, lam - float(z) * sd)
    hi = lam + float(z) * sd
    covered = (y >= lo) & (y <= hi)
    avg_width = float(np.mean(hi - lo))
    return {
        "coverage": float(np.mean(covered)),
        "avg_width": avg_width,
        "z": float(z),
    }


def rolling_origin_backtest(
    frame: pd.DataFrame,
    cat_cols: Tuple[str, ...],
    num_cols: Tuple[str, ...],
    test_days: int,
    folds: int,
    alpha: float,
    max_iter: int,
    half_life_days: float,
    min_weight: float,
) -> dict[str, object]:
    if int(folds) <= 0:
        return {"enabled": False, "folds": []}

    frame = frame.sort_values("date").copy()
    max_date = pd.Timestamp(frame["date"].max())
    folds_out = []
    for k in range(int(folds), 0, -1):
        test_end = max_date - pd.Timedelta(days=(k - 1) * int(test_days))
        test_start = test_end - pd.Timedelta(days=int(test_days))

        tr = frame[frame["date"] < test_start].copy()
        te = frame[(frame["date"] >= test_start) & (frame["date"] < test_end)].copy()
        if len(tr) < 20000 or len(te) < 1000:
            continue

        X_tr = tr[list(cat_cols) + list(num_cols)]
        X_te = te[list(cat_cols) + list(num_cols)]
        y_h_tr = tr["home_goals"].astype(int).to_numpy()
        y_a_tr = tr["away_goals"].astype(int).to_numpy()
        y_h_te = te["home_goals"].astype(int).to_numpy()
        y_a_te = te["away_goals"].astype(int).to_numpy()

        w = build_time_decay_weights(
            tr["date"],
            ref_date=pd.Timestamp(tr["date"].max()),
            half_life_days=half_life_days,
            min_weight=min_weight,
        )

        mh = build_pipeline(cat_cols, num_cols, alpha=float(alpha), max_iter=int(max_iter))
        ma = build_pipeline(cat_cols, num_cols, alpha=float(alpha), max_iter=int(max_iter))
        mh.fit(X_tr, y_h_tr, model__sample_weight=w)
        ma.fit(X_tr, y_a_tr, model__sample_weight=w)

        l_h = np.clip(mh.predict(X_te), 1e-6, None)
        l_a = np.clip(ma.predict(X_te), 1e-6, None)
        nll = neg_log_likelihood(y_h_te, y_a_te, l_h, l_a)
        folds_out.append(
            {
                "train_rows": int(len(tr)),
                "test_rows": int(len(te)),
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                "nll_ind": float(nll),
            }
        )

    nlls = [f["nll_ind"] for f in folds_out]
    return {
        "enabled": True,
        "n_folds": int(len(folds_out)),
        "nll_mean": float(np.mean(nlls)) if nlls else float("nan"),
        "nll_std": float(np.std(nlls, ddof=1)) if len(nlls) > 1 else float("nan"),
        "folds": folds_out,
    }


def choose_rho_with_significance(
    y_home: np.ndarray,
    y_away: np.ndarray,
    lam_home: np.ndarray,
    lam_away: np.ndarray,
    rho_candidate: float,
    z_critical: float = 1.96,
) -> tuple[float, float, dict[str, float | bool]]:
    """
    Statistically grounded fallback: select rho=0 unless rho_candidate improves
    per-match DC log loss significantly on the validation slice.
    """
    loss_candidate = per_match_neg_log_likelihood_dixon_coles(
        y_home,
        y_away,
        lam_home,
        lam_away,
        rho=float(rho_candidate),
    )
    loss_null = per_match_neg_log_likelihood_dixon_coles(
        y_home,
        y_away,
        lam_home,
        lam_away,
        rho=0.0,
    )

    diff = loss_candidate - loss_null
    valid = np.isfinite(diff)
    diff = diff[valid]

    if len(diff) < 2:
        selected_rho = 0.0
        selected_metric = float(np.mean(loss_null[np.isfinite(loss_null)]))
        return selected_rho, selected_metric, {
            "n": int(len(diff)),
            "mean_diff": float("nan"),
            "se_diff": float("nan"),
            "z_critical": float(z_critical),
            "significant": False,
            "selected_null": True,
        }

    mean_diff = float(np.mean(diff))
    se_diff = float(np.std(diff, ddof=1) / np.sqrt(len(diff)))
    significant = bool(mean_diff < (-float(z_critical) * max(se_diff, 1e-12)))

    if significant:
        selected_rho = float(rho_candidate)
        selected_metric = float(np.mean(loss_candidate[np.isfinite(loss_candidate)]))
        selected_null = False
    else:
        selected_rho = 0.0
        selected_metric = float(np.mean(loss_null[np.isfinite(loss_null)]))
        selected_null = True

    return selected_rho, selected_metric, {
        "n": int(len(diff)),
        "mean_diff": mean_diff,
        "se_diff": se_diff,
        "z_critical": float(z_critical),
        "significant": significant,
        "selected_null": selected_null,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", type=str, default=DEFAULT_MATCHES_REL_PATH)
    parser.add_argument("--stadiums", type=str, default=DEFAULT_STADIUMS_REL_PATH)
    parser.add_argument("--out", type=str, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--test-days", type=int, default=365, help="How many most-recent days to reserve for test.")
    parser.add_argument("--windows", type=int, nargs="+", default=[5, 10], help="Rolling windows for form features.")
    parser.add_argument("--max-goals", type=int, default=4, help="Used for app scoreline grid display.")
    parser.add_argument("--alpha", type=float, default=1e-4, help="Regularization for PoissonRegressor.")
    parser.add_argument("--low-score-alpha", type=float, default=0.0, help="Weight for low-score mixture calibration (0 disables).")
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--decay-half-life-days", type=float, default=0.0,
                        help="Half-life (days) for exponential time-decay sample weighting. 0 disables decay.")
    parser.add_argument("--tune-decay", action="store_true",
                        help="Auto-tune time-decay half-life on an internal validation slice.")
    parser.add_argument("--decay-candidates", type=float, nargs="+", default=[0.0, 365.0, 730.0, 1095.0],
                        help="Half-life candidates (days) evaluated when --tune-decay is enabled.")
    parser.add_argument("--val-days", type=int, default=180,
                        help="Validation slice length in days from end of train period for --tune-decay.")
    parser.add_argument("--tune-metric", choices=["ind_nll", "dc_nll"], default="ind_nll",
                        help="Validation metric used by --tune-decay.")
    parser.add_argument("--use-ewm-features", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable/disable EWM form features for ablation.")
    parser.add_argument("--use-adjusted-features", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable/disable opponent-adjusted form features for ablation.")
    parser.add_argument("--fit-dc", action="store_true", help="Fit Dixon-Coles low-score correction parameters.")
    parser.add_argument("--dc-rho-min", type=float, default=-0.30, help="Min rho for Dixon-Coles grid search.")
    parser.add_argument("--dc-rho-max", type=float, default=0.30, help="Max rho for Dixon-Coles grid search.")
    parser.add_argument("--dc-coarse-steps", type=int, default=121, help="Coarse grid size for Dixon-Coles rho search.")
    parser.add_argument("--dc-fine-steps", type=int, default=121, help="Fine grid size for Dixon-Coles rho search.")
    parser.add_argument("--dc-min-league-matches", type=int, default=2500, help="Minimum training matches required to fit per-league rho.")
    parser.add_argument("--dc-optimize-oot", action="store_true",
                        help="When tuning with --tune-metric dc_nll, optimize rho on the out-of-time validation slice and use tuned rho.")
    parser.add_argument("--dc-significance-z", type=float, default=1.96,
                        help="Critical z-value for selecting rho over 0 in OOT tuning (default 1.96 ~= 95% confidence).")
    parser.add_argument("--dc-max-top-share", type=float, default=0.0,
                        help="Optional heuristic cap for top-score mode share (0 disables; default disabled).")
    parser.add_argument("--fit-score-calibration", action=argparse.BooleanOptionalAction, default=True,
                        help="Fit post-hoc temperature calibration for exact-score distributions.")
    parser.add_argument("--calibration-val-days", type=int, default=180,
                        help="Validation window size (days, inside train split) for post-hoc scoreline calibration.")
    parser.add_argument("--score-calibration-by-league", action=argparse.BooleanOptionalAction, default=True,
                        help="Fit league-specific temperatures where enough calibration data exists.")
    parser.add_argument("--calibration-min-league-rows", type=int, default=1200,
                        help="Minimum validation rows to fit league-specific temperature.")
    parser.add_argument("--calibration-temperature-floor", type=float, default=1.0,
                        help="Minimum temperature after calibration (>=1.0 flattens).")
    parser.add_argument("--fit-low-score-mixture", action=argparse.BooleanOptionalAction, default=False,
                        help="Fit a minimal low-score mixture calibration on validation split (after temperature calibration).")
    parser.add_argument("--backtest-folds", type=int, default=0,
                        help="Optional rolling-origin backtest folds for stability diagnostics (0 disables).")
    args = parser.parse_args()

    if args.tune_metric == "dc_nll" and not args.fit_dc:
        raise ValueError("--tune-metric dc_nll requires --fit-dc")

    if float(args.dc_max_top_share) > 0.0 and not bool(args.fit_dc):
        print(
            "[warning] --dc-max-top-share is set but --fit-dc is disabled; "
            "the concentration guardrail only applies when Dixon-Coles is enabled."
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / DEFAULT_ARTIFACT_FILENAME

    config = FeatureConfig(
        windows=tuple(args.windows),
        max_goals=int(args.max_goals),
        use_travel_distance=True,
        use_ewm_features=bool(args.use_ewm_features),
        use_adjusted_features=bool(args.use_adjusted_features),
    )

    print("Loading data...")
    matches = load_matches_csv(args.matches)
    stadiums = load_stadiums_csv(args.stadiums) if args.stadiums else None

    print("Building training frame (this can take a bit on first run)...")
    frame = build_training_frame(matches, stadiums, config)

    # Only train on rows with observed goals
    frame = frame.dropna(subset=["home_goals", "away_goals"]).reset_index(drop=True)

    # Clean time-based split
    max_date = frame["date"].max()
    cutoff = max_date - pd.Timedelta(days=int(args.test_days))
    train_df = frame[frame["date"] < cutoff].copy()
    test_df = frame[frame["date"] >= cutoff].copy()

    print(f"Max date in data:  {max_date.date()}")
    print(f"Test cutoff date:  {cutoff.date()}  (test has matches on/after this date)")
    print(f"Train rows: {len(train_df):,}")
    print(f"Test rows:  {len(test_df):,}")
    print(f"Feature toggles: use_ewm={config.use_ewm_features}, use_adjusted={config.use_adjusted_features}")

    cat_cols = model_categorical_columns()
    num_cols = model_numeric_columns(config)

    decay_tuning: Optional[Dict[str, object]] = None
    selected_half_life_days = float(args.decay_half_life_days)
    selected_rho_from_tuning: Optional[float] = None
    if bool(args.tune_decay):
        print("\n[Tuning] Running decay half-life tuning...")
        decay_tuning = tune_decay_half_life(
            train_df=train_df,
            cat_cols=cat_cols,
            num_cols=num_cols,
            candidates=[float(c) for c in args.decay_candidates],
            val_days=int(args.val_days),
            alpha=float(args.alpha),
            max_iter=int(args.max_iter),
            tune_metric=str(args.tune_metric),
            fit_dc=bool(args.fit_dc),
            dc_rho_min=float(args.dc_rho_min),
            dc_rho_max=float(args.dc_rho_max),
            dc_coarse_steps=int(args.dc_coarse_steps),
            dc_fine_steps=int(args.dc_fine_steps),
            dc_optimize_oot=bool(args.dc_optimize_oot),
            dc_significance_z=float(args.dc_significance_z),
        )
        selected_half_life_days = float(decay_tuning.get("best_half_life_days", selected_half_life_days))
        if decay_tuning.get("best_rho") is not None:
            selected_rho_from_tuning = float(decay_tuning["best_rho"])
        if "warning" in decay_tuning:
            print(f"[Tuning] {decay_tuning['warning']}")
        else:
            print(
                f"[Tuning] Selected half-life={selected_half_life_days:.1f} days "
                f"using {decay_tuning.get('metric_name', args.tune_metric)}"
            )
            if selected_rho_from_tuning is not None:
                print(
                    f"[Tuning] Selected rho={selected_rho_from_tuning:.4f} "
                    f"({decay_tuning.get('best_rho_source', 'unknown')})"
                )
    else:
        print(f"[Tuning] Using fixed half-life={selected_half_life_days:.1f} days")

    # Defensive check: ensure columns exist
    missing = [c for c in list(cat_cols) + list(num_cols) if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")

    X_train = train_df[list(cat_cols) + list(num_cols)]
    X_test = test_df[list(cat_cols) + list(num_cols)]

    y_home_train = train_df["home_goals"].astype(int).to_numpy()
    y_away_train = train_df["away_goals"].astype(int).to_numpy()
    y_home_test = test_df["home_goals"].astype(int).to_numpy()
    y_away_test = test_df["away_goals"].astype(int).to_numpy()
    div_test = test_df["div"].astype(str).to_numpy()

    ref_date_train_max = pd.Timestamp(train_df["date"].max())
    min_weight = 1e-6
    train_weights = build_time_decay_weights(
        train_df["date"],
        ref_date=ref_date_train_max,
        half_life_days=selected_half_life_days,
        min_weight=min_weight,
    )
    if selected_half_life_days > 0:
        print(
            f"Time-decay: half_life_days={selected_half_life_days:.1f}, "
            f"weight_range=[{float(np.min(train_weights)):.6f}, {float(np.max(train_weights)):.6f}]"
        )
    else:
        print("Time-decay: disabled (equal weighting)")

    print("\nTraining home-goals model...")
    model_home = build_pipeline(cat_cols, num_cols, alpha=float(args.alpha), max_iter=int(args.max_iter))
    model_home.fit(X_train, y_home_train, model__sample_weight=train_weights)

    print("Training away-goals model...")
    model_away = build_pipeline(cat_cols, num_cols, alpha=float(args.alpha), max_iter=int(args.max_iter))
    model_away.fit(X_train, y_away_train, model__sample_weight=train_weights)

    home_coef = np.asarray(model_home.named_steps["model"].coef_)
    away_coef = np.asarray(model_away.named_steps["model"].coef_)
    home_nonzero = int((np.abs(home_coef) > 1e-12).sum())
    away_nonzero = int((np.abs(away_coef) > 1e-12).sum())
    print(f"Non-zero coefficients (home/away): {home_nonzero} / {away_nonzero}")
    if home_nonzero == 0 and away_nonzero == 0:
        raise RuntimeError(
            "Both Poisson models collapsed to intercept-only (all coefficients are zero). "
            "Try increasing --max-iter (e.g. 1000) and verify feature values are finite."
        )

    # In-sample lambdas for low-score correlation calibration
    lam_home_train = np.clip(model_home.predict(X_train), 1e-6, None)
    lam_away_train = np.clip(model_away.predict(X_train), 1e-6, None)

    # optionally compute low-score empirical mixture settings used in calibration
    low_score_cfg: Dict[str, object] = {}
    if float(args.low_score_alpha) > 0.0:
        summary = low_score_calibration_summary(
            y_home_train, y_away_train, lam_home_train, lam_away_train, rho=None
        )
        # target empirical probabilities for the four low scores
        target = {score: summary[score]["empirical"] for score in summary}
        low_score_cfg = {
            "enabled": True,
            "alpha": float(args.low_score_alpha),
            "target_probs": target,
        }
        print(f"Low-score calibration: alpha={low_score_cfg['alpha']} target={target}")

    rho_global = None
    rho_global_train_fit = None
    rho_global_source = None
    train_nll_dc = None
    rho_by_div = {}

    if args.fit_dc:
        rho_global_train_fit, train_nll_dc = fit_dixon_coles_rho(
            y_home_train,
            y_away_train,
            lam_home_train,
            lam_away_train,
            rho_min=float(args.dc_rho_min),
            rho_max=float(args.dc_rho_max),
            coarse_steps=int(args.dc_coarse_steps),
            fine_steps=int(args.dc_fine_steps),
            sample_weight=train_weights,
        )
        rho_global = float(rho_global_train_fit)
        rho_global_source = "train_fit"

        if (
            bool(args.tune_decay)
            and str(args.tune_metric) == "dc_nll"
            and bool(args.dc_optimize_oot)
            and selected_rho_from_tuning is not None
        ):
            rho_global = float(selected_rho_from_tuning)
            rho_global_source = "oot_tuned"

        for div, div_idx in train_df.groupby("div").groups.items():
            idx = np.asarray(list(div_idx), dtype=int)
            if len(idx) < int(args.dc_min_league_matches):
                continue

            rho_div, _ = fit_dixon_coles_rho(
                y_home_train[idx],
                y_away_train[idx],
                lam_home_train[idx],
                lam_away_train[idx],
                rho_min=float(args.dc_rho_min),
                rho_max=float(args.dc_rho_max),
                coarse_steps=int(args.dc_coarse_steps),
                fine_steps=int(args.dc_fine_steps),
                sample_weight=train_weights[idx],
            )
            rho_by_div[str(div)] = float(rho_div)

    # Evaluate
    lam_home = np.clip(model_home.predict(X_test), 1e-6, None)
    lam_away = np.clip(model_away.predict(X_test), 1e-6, None)

    scoreline_calibration = {
        "enabled": False,
        "method": "temperature",
        "temperature": 1.0,
        "nll_uncal": float("nan"),
        "nll_cal": float("nan"),
    }

    train_sorted = train_df.sort_values("date").copy()
    calib_cutoff = pd.Timestamp(train_sorted["date"].max()) - pd.Timedelta(days=int(args.calibration_val_days))
    fit_cal_df = train_sorted[train_sorted["date"] < calib_cutoff].copy()
    val_cal_df = train_sorted[train_sorted["date"] >= calib_cutoff].copy()

    have_calibration_rows = len(fit_cal_df) >= 5000 and len(val_cal_df) >= 1000

    y_home_val_cal = np.array([], dtype=int)
    y_away_val_cal = np.array([], dtype=int)
    lam_home_val_cal = np.array([], dtype=float)
    lam_away_val_cal = np.array([], dtype=float)
    div_val_cal = np.array([], dtype=str)
    rho_cal = float(rho_global) if rho_global is not None else None

    if have_calibration_rows:
        X_fit_cal = fit_cal_df[list(cat_cols) + list(num_cols)]
        X_val_cal = val_cal_df[list(cat_cols) + list(num_cols)]
        y_home_fit_cal = fit_cal_df["home_goals"].astype(int).to_numpy()
        y_away_fit_cal = fit_cal_df["away_goals"].astype(int).to_numpy()
        y_home_val_cal = val_cal_df["home_goals"].astype(int).to_numpy()
        y_away_val_cal = val_cal_df["away_goals"].astype(int).to_numpy()
        div_val_cal = val_cal_df["div"].astype(str).to_numpy()

        ref_date_fit_cal = pd.Timestamp(fit_cal_df["date"].max())
        w_fit_cal = build_time_decay_weights(
            fit_cal_df["date"],
            ref_date=ref_date_fit_cal,
            half_life_days=selected_half_life_days,
            min_weight=min_weight,
        )

        model_home_cal = build_pipeline(cat_cols, num_cols, alpha=float(args.alpha), max_iter=int(args.max_iter))
        model_away_cal = build_pipeline(cat_cols, num_cols, alpha=float(args.alpha), max_iter=int(args.max_iter))
        model_home_cal.fit(X_fit_cal, y_home_fit_cal, model__sample_weight=w_fit_cal)
        model_away_cal.fit(X_fit_cal, y_away_fit_cal, model__sample_weight=w_fit_cal)

        lam_home_val_cal = np.clip(model_home_cal.predict(X_val_cal), 1e-6, None)
        lam_away_val_cal = np.clip(model_away_cal.predict(X_val_cal), 1e-6, None)

    if bool(args.fit_score_calibration):
        if have_calibration_rows:
            scoreline_calibration = fit_temperature_calibration(
                y_home=y_home_val_cal,
                y_away=y_away_val_cal,
                lam_home=lam_home_val_cal,
                lam_away=lam_away_val_cal,
                max_goals=int(config.max_goals),
                rho=rho_cal,
            )
            # enforce a minimum (floor) on the fitted temperature if requested
            t_floor = float(args.calibration_temperature_floor)
            if t_floor > 1.0 and scoreline_calibration.get("temperature", 1.0) < t_floor:
                scoreline_calibration["temperature"] = t_floor

            if bool(args.score_calibration_by_league):
                temps_by_div: Dict[str, float] = {}
                for div_name, idx in val_cal_df.groupby("div").groups.items():
                    idx_arr = np.asarray(list(idx), dtype=int)
                    if len(idx_arr) < int(args.calibration_min_league_rows):
                        continue
                    fit_div = fit_temperature_calibration(
                        y_home=y_home_val_cal[idx_arr],
                        y_away=y_away_val_cal[idx_arr],
                        lam_home=lam_home_val_cal[idx_arr],
                        lam_away=lam_away_val_cal[idx_arr],
                        max_goals=int(config.max_goals),
                        rho=rho_cal,
                    )
                    temps_by_div[str(div_name)] = float(fit_div.get("temperature", 1.0))
                if temps_by_div:
                    scoreline_calibration["temperatures_by_div"] = temps_by_div

    # incorporate low-score mixture if configured earlier
    if low_score_cfg:
        scoreline_calibration["low_score_mixture"] = low_score_cfg

    scoreline_calibration["split_cutoff"] = str(calib_cutoff.date())
    scoreline_calibration["val_rows"] = int(len(val_cal_df))

    reliability = {"ece": float("nan"), "enabled": False, "reason": "not_enough_calibration_rows"}
    event_reliability = {"events": {}, "enabled": False, "reason": "not_enough_calibration_rows"}

    if have_calibration_rows:
        reliability = top1_reliability_report(
            y_home=y_home_val_cal,
            y_away=y_away_val_cal,
            lam_home=lam_home_val_cal,
            lam_away=lam_away_val_cal,
            max_goals=int(config.max_goals),
            rho=rho_cal,
            calibration_cfg=scoreline_calibration,
            n_bins=10,
            divs=div_val_cal,
        )
        event_reliability = event_level_reliability_report(
            y_home=y_home_val_cal,
            y_away=y_away_val_cal,
            lam_home=lam_home_val_cal,
            lam_away=lam_away_val_cal,
            max_goals=int(config.max_goals),
            rho=rho_cal,
            calibration_cfg=scoreline_calibration,
            events=("1-1", "1-0", "0-1", "0-0"),
            n_bins=10,
            divs=div_val_cal,
        )

    scoreline_calibration["top1_reliability"] = reliability
    scoreline_calibration["event_reliability"] = event_reliability

    if bool(args.fit_low_score_mixture) and have_calibration_rows:
        low_mix = fit_low_score_mixture_calibration(
            y_home=y_home_val_cal,
            y_away=y_away_val_cal,
            lam_home=lam_home_val_cal,
            lam_away=lam_away_val_cal,
            max_goals=int(config.max_goals),
            rho=rho_cal,
            calibration_cfg=scoreline_calibration,
        )
        scoreline_calibration["low_score_mixture"] = low_mix
        print(
            "Low-score mixture calibration: "
            f"enabled={low_mix.get('enabled', False)}, "
            f"alpha={float(low_mix.get('alpha', 0.0)):.3f}, "
            f"NLL {float(low_mix.get('nll_base', np.nan)):.4f} -> {float(low_mix.get('nll_mix', np.nan)):.4f}"
        )

    print(
        f"Scoreline calibration (temperature): T={scoreline_calibration['temperature']:.3f}, "
        f"NLL {float(scoreline_calibration.get('nll_uncal', float('nan'))):.4f} -> {float(scoreline_calibration.get('nll_cal', float('nan'))):.4f}, "
        f"ECE={reliability['ece']:.4f}"
    )
    ev = event_reliability.get("events", {})
    if ev:
        print("Event reliability (ECE / mean_pred vs prevalence):")
        for event in ("1-1", "1-0", "0-1", "0-0"):
            if event not in ev:
                continue
            row = ev[event]
            print(
                f"  {event}: ECE={row['ece']:.4f}, "
                f"pred={row['mean_pred']:.2%}, emp={row['prevalence']:.2%}"
            )
    else:
        print("Scoreline calibration: skipped (not enough calibration rows)")

    rho_guardrail = None
    if rho_global is not None and float(args.dc_max_top_share) > 0:
        mode_share_before = top_score_mode_share(
            lam_home,
            lam_away,
            max_goals=int(config.max_goals),
            rho=float(rho_global),
            calibration_cfg=scoreline_calibration,
            divs=div_test,
        )
        if mode_share_before > float(args.dc_max_top_share):
            rho_adj, mode_share_after = shrink_rho_for_concentration(
                lam_home,
                lam_away,
                rho=float(rho_global),
                max_goals=int(config.max_goals),
                max_top_share=float(args.dc_max_top_share),
                calibration_cfg=scoreline_calibration,
                divs=div_test,
            )
            rho_guardrail = {
                "enabled": True,
                "max_top_share": float(args.dc_max_top_share),
                "mode_share_before": float(mode_share_before),
                "mode_share_after": float(mode_share_after),
                "rho_before": float(rho_global),
                "rho_after": float(rho_adj),
            }
            rho_global = float(rho_adj)
            rho_global_source = f"{rho_global_source}+guardrail"

    # Regression-style metrics on expected goals
    metrics = expected_goals_regression_metrics(y_home_test, y_away_test, lam_home, lam_away)
    nll = neg_log_likelihood(y_home_test, y_away_test, lam_home, lam_away)
    nll_dc = None
    if rho_global is not None:
        nll_dc = neg_log_likelihood_dixon_coles(y_home_test, y_away_test, lam_home, lam_away, float(rho_global))

    # Scientific benchmark panel + DM tests
    bench = benchmark_panel(
        y_home_train=y_home_train,
        y_away_train=y_away_train,
        y_home_test=y_home_test,
        y_away_test=y_away_test,
        model_top1_acc=0.0,
        model_nll_ind=float(nll),
    )

    train_pairs = list(zip(y_home_train.astype(int), y_away_train.astype(int)))
    train_counts: Dict[tuple[int, int], int] = {}
    for pair in train_pairs:
        train_counts[pair] = train_counts.get(pair, 0) + 1
    total_train = max(len(train_pairs), 1)
    probs_emp = {k: v / total_train for k, v in train_counts.items()}
    eps = 1e-12
    loss_model = []
    loss_emp = []
    for yh_i, ya_i, lh_i, la_i in zip(y_home_test, y_away_test, lam_home, lam_away):
        loss_model.append(-(
            yh_i * np.log(max(lh_i, eps)) - lh_i - math.lgamma(int(yh_i) + 1)
            + ya_i * np.log(max(la_i, eps)) - la_i - math.lgamma(int(ya_i) + 1)
        ))
        loss_emp.append(-np.log(max(probs_emp.get((int(yh_i), int(ya_i)), eps), eps)))
    dm_emp = diebold_mariano_test(np.asarray(loss_model), np.asarray(loss_emp))

    lambda_pi = {
        "home": lambda_interval_diagnostics(y_home_test, lam_home),
        "away": lambda_interval_diagnostics(y_away_test, lam_away),
    }

    rolling_bt = rolling_origin_backtest(
        frame=frame,
        cat_cols=cat_cols,
        num_cols=num_cols,
        test_days=int(args.test_days),
        folds=int(args.backtest_folds),
        alpha=float(args.alpha),
        max_iter=int(args.max_iter),
        half_life_days=float(selected_half_life_days),
        min_weight=float(min_weight),
    )
    acc = evaluation_accuracy_summary(
        y_home_test,
        y_away_test,
        lam_home,
        lam_away,
        max_goals=int(config.max_goals),
        rho=float(rho_global) if rho_global is not None else None,
        calibration_cfg=scoreline_calibration,
        divs=div_test,
    )
    top_score_diag = top_score_frequency_distribution(
        lam_home,
        lam_away,
        max_goals=int(config.max_goals),
        rho=float(rho_global) if rho_global is not None else None,
        top_n=8,
        calibration_cfg=scoreline_calibration,
        divs=div_test,
    )

    print("\n=== Test metrics ===")
    print(f"Home goals MAE:   {metrics['home_mae']:.4f}    RMSE: {metrics['home_rmse']:.4f}    Poisson dev: {metrics['home_dev']:.4f}")
    print(f"Away goals MAE:   {metrics['away_mae']:.4f}    RMSE: {metrics['away_rmse']:.4f}    Poisson dev: {metrics['away_dev']:.4f}")
    print(f"Exact score top-1 accuracy:  {acc['exact_score_top1_acc']:.2%}")
    print(f"W/D/L accuracy:              {acc['wdl_acc']:.2%}")
    print(f"Avg NLL (score):  {nll:.4f}  (independent Poisson)")
    if rho_global is not None:
        print(f"Dixon-Coles rho (global):  {rho_global:.4f}")
        print(f"Dixon-Coles rho source:    {rho_global_source}")
        if rho_global_train_fit is not None and rho_global_source != "train_fit":
            print(f"Dixon-Coles rho (train):   {float(rho_global_train_fit):.4f}")
        if rho_guardrail is not None:
            print(
                "DC concentration guardrail: "
                f"share {rho_guardrail['mode_share_before']:.2%} -> {rho_guardrail['mode_share_after']:.2%}, "
                f"rho {rho_guardrail['rho_before']:.4f} -> {rho_guardrail['rho_after']:.4f}"
            )
        print(f"Avg NLL (DC):              {nll_dc:.4f}  (low-score corrected)")
        print(f"Train NLL (DC):            {train_nll_dc:.4f}")
        print(f"Per-league rho fitted:     {len(rho_by_div)} leagues")
    else:
        print("Dixon-Coles fitting:       disabled (use --fit-dc to enable)")

    bench["model_top1_acc"] = float(acc["exact_score_top1_acc"])
    print("\nBenchmark panel (test split):")
    print(
        f"  model_top1={bench['model_top1_acc']:.2%} | "
        f"always_train_mode={bench['always_train_mode_acc']:.2%} | "
        f"always_1_1={bench['always_1_1_acc']:.2%} | "
        f"uniform_random={bench['uniform_random_acc']:.2%}"
    )
    print(
        f"  model_nll={bench['model_nll_ind']:.4f} | empirical_nll={bench['empirical_nll']:.4f} | "
        f"DM z={dm_emp['z']:.3f}, p={dm_emp['p_value']:.4f}"
    )
    print(
        f"Lambda PI coverage (95%): home={lambda_pi['home']['coverage']:.2%}, "
        f"away={lambda_pi['away']['coverage']:.2%}"
    )
    if rolling_bt.get("enabled") and rolling_bt.get("n_folds", 0) > 0:
        print(
            f"Rolling backtest: folds={rolling_bt['n_folds']}, "
            f"mean NLL={rolling_bt['nll_mean']:.4f}, std={rolling_bt['nll_std']:.4f}"
        )

    print("Top predicted score frequencies (test split):")
    for row in top_score_diag["rows"]:
        print(f"  {row['score']}: {row['count']} ({row['share']:.2%})")

    low_score = low_score_calibration_summary(
        y_home_test,
        y_away_test,
        lam_home,
        lam_away,
        rho=float(rho_global) if rho_global is not None else None,
    )
    mode_label = "Dixon-Coles" if rho_global is not None else "Independent Poisson"
    print(f"\nLow-score calibration ({mode_label}, test split):")
    for score in ("0-0", "1-0", "0-1", "1-1"):
        row = low_score[score]
        print(
            f"  {score}: empirical={row['empirical']:.2%}  predicted={row['predicted']:.2%}  delta={row['delta_pp']:+.2f} pp"
        )

    # Append run summary to CSV log for run-to-run tracking
    calibration_log_path = out_dir / "low_score_calibration_log.csv"
    log_row = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "fit_dc": bool(args.fit_dc),
        "dc_mode": mode_label,
        "rho_global": float(rho_global) if rho_global is not None else np.nan,
        "test_rows": int(len(test_df)),
        "nll_independent": float(nll),
        "nll_dc": float(nll_dc) if nll_dc is not None else np.nan,
    }
    for score in ("0-0", "1-0", "0-1", "1-1"):
        log_row[f"{score}_empirical"] = float(low_score[score]["empirical"])
        log_row[f"{score}_predicted"] = float(low_score[score]["predicted"])
        log_row[f"{score}_delta_pp"] = float(low_score[score]["delta_pp"])

    pd.DataFrame([log_row]).to_csv(
        calibration_log_path,
        mode="a",
        index=False,
        header=not calibration_log_path.exists(),
    )
    print(f"Saved calibration log row to: {calibration_log_path}")

    strict_scientific_mode = bool(
        args.fit_dc
        and args.tune_decay
        and str(args.tune_metric) == "dc_nll"
        and args.dc_optimize_oot
        and bool(args.fit_score_calibration)
        and float(args.dc_max_top_share) == 0.0
    )

    artifact = {
        "model_home": model_home,
        "model_away": model_away,
        "config": {
            "windows": list(config.windows),
            "max_goals": int(config.max_goals),
            "use_travel_distance": bool(config.use_travel_distance),
            "ewm_span": int(config.ewm_span),
            "use_ewm_features": bool(config.use_ewm_features),
            "use_adjusted_features": bool(config.use_adjusted_features),
        },
        "cat_cols": list(cat_cols),
        "num_cols": list(num_cols),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "cutoff_date": str(cutoff.date()),
        "max_date": str(max_date.date()),
        "dixon_coles_rho": float(rho_global) if rho_global is not None else None,
        "dixon_coles": {
            "rho_global": float(rho_global) if rho_global is not None else None,
            "rho_global_source": rho_global_source,
            "rho_global_train_fit": float(rho_global_train_fit) if rho_global_train_fit is not None else None,
            "rho_guardrail": rho_guardrail,
            "rho_by_div": rho_by_div,
            "rho_search_bounds": {
                "min": float(args.dc_rho_min),
                "max": float(args.dc_rho_max),
            },
            "normalized": True,
            "min_league_matches": int(args.dc_min_league_matches),
            "enabled": bool(args.fit_dc),
        },
        "time_decay": {
            "half_life_days": float(selected_half_life_days),
            "ref_date_train_max": str(ref_date_train_max.date()),
            "min_weight": float(min_weight),
            "weight_min": float(np.min(train_weights)),
            "weight_max": float(np.max(train_weights)),
            "weight_mean": float(np.mean(train_weights)),
        },
        "decay_tuning": {
            "enabled": bool(args.tune_decay),
            "metric": str(args.tune_metric),
            "dc_optimize_oot": bool(args.dc_optimize_oot),
            "dc_significance_z": float(args.dc_significance_z),
            "candidates": [float(c) for c in args.decay_candidates],
            "val_days": int(args.val_days),
            "selected_half_life_days": float(selected_half_life_days),
            "selected_rho": float(selected_rho_from_tuning) if selected_rho_from_tuning is not None else None,
            "summary": decay_tuning,
        },
        "scientific_mode": {
            "strict_scientific_mode": strict_scientific_mode,
            "criteria": {
                "fit_dc": bool(args.fit_dc),
                "tune_decay": bool(args.tune_decay),
                "tune_metric": str(args.tune_metric),
                "dc_optimize_oot": bool(args.dc_optimize_oot),
                "dc_significance_z": float(args.dc_significance_z),
                "dc_max_top_share": float(args.dc_max_top_share),
                "fit_score_calibration": bool(args.fit_score_calibration),
            },
        },
        "training_environment": {
            "python_version": sys.version.split()[0],
            "numpy_version": _installed_version("numpy"),
            "pandas_version": _installed_version("pandas"),
            "scikit_learn_version": _installed_version("scikit-learn"),
        },
        "scoreline_calibration": scoreline_calibration,
        "diagnostics": {
            "top_score_frequency_test": top_score_diag,
            "benchmark_panel": bench,
            "diebold_mariano": {
                "model_vs_empirical": dm_emp,
            },
            "lambda_interval": lambda_pi,
            "rolling_backtest": rolling_bt,
        },
    }

    print(f"\nSaving artifact to: {artifact_path}")
    joblib.dump(artifact, artifact_path)

    print("\nDone.")
    print("Next:")
    print("  1) streamlit run app.py")
    print("  2) Pick teams/league, enter odds, and predict scorelines.")


if __name__ == "__main__":
    main()
