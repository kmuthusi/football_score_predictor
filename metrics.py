from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error

from predict import dixon_coles_tau


def _log_poisson_pmf(k: int, lam: float) -> float:
    lam = float(max(lam, 1e-12))
    return -lam + k * math.log(lam) - math.lgamma(k + 1)


def neg_log_likelihood(
    y_home: np.ndarray,
    y_away: np.ndarray,
    lam_home: np.ndarray,
    lam_away: np.ndarray,
) -> float:
    total = 0.0
    n = len(y_home)
    for i in range(n):
        total += -(_log_poisson_pmf(int(y_home[i]), float(lam_home[i])) + _log_poisson_pmf(int(y_away[i]), float(lam_away[i])))
    return total / max(n, 1)


def neg_log_likelihood_dixon_coles(
    y_home: np.ndarray,
    y_away: np.ndarray,
    lam_home: np.ndarray,
    lam_away: np.ndarray,
    rho: float,
    sample_weight: np.ndarray | None = None,
) -> float:
    total = 0.0
    total_w = 0.0
    n = len(y_home)
    for i in range(n):
        y_h = int(y_home[i])
        y_a = int(y_away[i])
        l_h = float(lam_home[i])
        l_a = float(lam_away[i])
        w = float(sample_weight[i]) if sample_weight is not None else 1.0

        tau = dixon_coles_tau(y_h, y_a, l_h, l_a, rho)
        if tau <= 0:
            return float("inf")

        total += w * (-(_log_poisson_pmf(y_h, l_h) + _log_poisson_pmf(y_a, l_a) + math.log(tau)))
        total_w += w
    return total / max(total_w, 1e-12)


def per_match_neg_log_likelihood_dixon_coles(
    y_home: np.ndarray,
    y_away: np.ndarray,
    lam_home: np.ndarray,
    lam_away: np.ndarray,
    rho: float,
) -> np.ndarray:
    n = len(y_home)
    out = np.zeros(n, dtype=float)
    for i in range(n):
        y_h = int(y_home[i])
        y_a = int(y_away[i])
        l_h = float(lam_home[i])
        l_a = float(lam_away[i])
        tau = dixon_coles_tau(y_h, y_a, l_h, l_a, rho)
        if tau <= 0:
            out[i] = float("inf")
            continue
        out[i] = -(_log_poisson_pmf(y_h, l_h) + _log_poisson_pmf(y_a, l_a) + math.log(tau))
    return out


def _grid_search_rho(
    y_home: np.ndarray,
    y_away: np.ndarray,
    lam_home: np.ndarray,
    lam_away: np.ndarray,
    rho_min: float,
    rho_max: float,
    steps: int,
    sample_weight: np.ndarray | None = None,
) -> Tuple[float, float]:
    best_rho = 0.0
    if sample_weight is not None:
        baseline = (
            np.asarray(sample_weight, dtype=float)
            * (-(
                np.vectorize(_log_poisson_pmf)(y_home.astype(int), lam_home.astype(float))
                + np.vectorize(_log_poisson_pmf)(y_away.astype(int), lam_away.astype(float))
            ))
        )
        best_nll = float(np.sum(baseline) / max(float(np.sum(sample_weight)), 1e-12))
    else:
        best_nll = neg_log_likelihood(y_home, y_away, lam_home, lam_away)

    grid = np.linspace(float(rho_min), float(rho_max), int(steps))
    for rho in grid:
        nll = neg_log_likelihood_dixon_coles(y_home, y_away, lam_home, lam_away, float(rho), sample_weight=sample_weight)
        if nll < best_nll:
            best_nll = nll
            best_rho = float(rho)
    return best_rho, best_nll


def fit_dixon_coles_rho(
    y_home: np.ndarray,
    y_away: np.ndarray,
    lam_home: np.ndarray,
    lam_away: np.ndarray,
    rho_min: float = -0.30,
    rho_max: float = 0.30,
    coarse_steps: int = 121,
    fine_steps: int = 121,
    sample_weight: np.ndarray | None = None,
) -> Tuple[float, float]:
    coarse_rho, coarse_nll = _grid_search_rho(
        y_home, y_away, lam_home, lam_away,
        rho_min=float(rho_min), rho_max=float(rho_max), steps=int(coarse_steps),
        sample_weight=sample_weight,
    )

    if int(fine_steps) <= 1:
        return coarse_rho, coarse_nll

    span = float(rho_max) - float(rho_min)
    fine_half_width = max(span / 10.0, 1e-6)
    fine_min = max(float(rho_min), coarse_rho - fine_half_width)
    fine_max = min(float(rho_max), coarse_rho + fine_half_width)
    return _grid_search_rho(
        y_home, y_away, lam_home, lam_away,
        rho_min=fine_min, rho_max=fine_max, steps=int(fine_steps),
        sample_weight=sample_weight,
    )


def expected_goals_regression_metrics(
    y_home: np.ndarray,
    y_away: np.ndarray,
    lam_home: np.ndarray,
    lam_away: np.ndarray,
) -> Dict[str, float]:
    return {
        "home_mae": float(mean_absolute_error(y_home, lam_home)),
        "away_mae": float(mean_absolute_error(y_away, lam_away)),
        "home_rmse": float(math.sqrt(mean_squared_error(y_home, lam_home))),
        "away_rmse": float(math.sqrt(mean_squared_error(y_away, lam_away))),
        "home_dev": float(mean_poisson_deviance(y_home, lam_home)),
        "away_dev": float(mean_poisson_deviance(y_away, lam_away)),
    }

def fit_isotonic_calibration(
    probs_flat: np.ndarray,
    targets_flat: np.ndarray,
) -> Dict[str, object]:
    """
    Fit isotonic regression calibration.
    More flexible than temperature scaling; handles non-linear miscalibration.
    
    Args:
        probs_flat: flattened predicted probabilities
        targets_flat: flattened binary targets (0 or 1)
    
    Returns:
        dict with isotonic_regressor (sklearn object) and diagnostics
    """
    iso_cal = IsotonicRegression(out_of_bounds='clip')
    iso_cal.fit(probs_flat, targets_flat)
    
    # Evaluate on same data (in practice, should be validation set)
    probs_cal = iso_cal.predict(probs_flat)
    
    # Compute NLL before/after
    eps = 1e-12
    nll_uncal = -np.mean(targets_flat * np.log(np.clip(probs_flat, eps, 1.0)) + 
                          (1 - targets_flat) * np.log(np.clip(1 - probs_flat, eps, 1.0)))
    nll_cal = -np.mean(targets_flat * np.log(np.clip(probs_cal, eps, 1.0)) + 
                        (1 - targets_flat) * np.log(np.clip(1 - probs_cal, eps, 1.0)))
    
    return {
        "enabled": True,
        "method": "isotonic",
        "nll_uncal": float(nll_uncal),
        "nll_cal": float(nll_cal),
        "nll_improvement": float(nll_uncal - nll_cal),
        "n_rows": int(len(probs_flat)),
        "isotonic_regressor": iso_cal,
    }