"""
predict.py

Model loading + probability utilities.

The model predicts:
  - expected home goals (lambda_home)
  - expected away goals (lambda_away)

Then we convert expected goals into a correct-score probability distribution
(using an independent Poisson assumption as a baseline).

"""

from __future__ import annotations

import math
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd


def apply_temperature_scaling(probs: np.ndarray, temperature: float) -> np.ndarray:
    """
    Apply temperature scaling to a probability vector and renormalize.
    T < 1 sharpens; T > 1 smooths.
    """
    p = np.asarray(probs, dtype=float)
    t = float(max(temperature, 1e-6))
    p = np.clip(p, 1e-12, None)
    p = np.power(p, 1.0 / t)
    z = float(p.sum())
    if z <= 0:
        return np.full_like(p, 1.0 / max(len(p), 1), dtype=float)
    return p / z


def calibrate_scoreline_matrix(
    mat: pd.DataFrame,
    calibration_cfg: Dict[str, Any] | None,
    div: str | None = None,
) -> pd.DataFrame:
    """
    Apply post-hoc calibration to a scoreline probability matrix.
    Supported: temperature scaling on flattened matrix probabilities.
    """
    if not calibration_cfg:
        return mat

    arr = mat.to_numpy(dtype=float)

    if bool(calibration_cfg.get("enabled", False)) and calibration_cfg.get("method") == "temperature":
        temp = float(calibration_cfg.get("temperature", 1.0))
        by_div = calibration_cfg.get("temperatures_by_div", {})
        if div is not None and isinstance(by_div, dict) and str(div) in by_div:
            temp = float(by_div[str(div)])
        flat = arr.reshape(-1)
        flat_cal = apply_temperature_scaling(flat, temperature=temp)
        arr = flat_cal.reshape(arr.shape)

    low_mix_cfg = calibration_cfg.get("low_score_mixture", {})
    if isinstance(low_mix_cfg, dict) and bool(low_mix_cfg.get("enabled", False)):
        arr = apply_low_score_mixture(arr, labels=list(mat.index), cfg=low_mix_cfg)

    return pd.DataFrame(arr, index=mat.index, columns=mat.columns)


def apply_low_score_mixture(arr: np.ndarray, labels: list[str], cfg: Dict[str, Any]) -> np.ndarray:
    """
    Blend base scoreline distribution with a low-score target distribution:
      p_mix = (1-a) * p + a * q

    q keeps configured event probabilities fixed and redistributes remaining mass
    over non-event cells proportional to base probabilities.
    """
    out = np.asarray(arr, dtype=float).copy()
    total = float(out.sum())
    if total <= 0:
        return out
    out = out / total

    alpha = float(cfg.get("alpha", 0.0))
    alpha = min(max(alpha, 0.0), 1.0)
    if alpha <= 0:
        return out

    target_probs = cfg.get("target_probs", {})
    if not isinstance(target_probs, dict) or not target_probs:
        return out

    label_to_idx = {str(label): i for i, label in enumerate(labels)}
    event_positions: list[tuple[int, int, float]] = []
    for score, prob in target_probs.items():
        if not isinstance(score, str) or "-" not in score:
            continue
        h, a = score.split("-", 1)
        if h in label_to_idx and a in label_to_idx:
            p = float(prob)
            if p >= 0:
                event_positions.append((label_to_idx[h], label_to_idx[a], p))

    if not event_positions:
        return out

    emp_sum = float(sum(p for _, _, p in event_positions))
    if emp_sum <= 0:
        return out
    if emp_sum > 1.0:
        scale = 1.0 / emp_sum
        event_positions = [(r, c, p * scale) for r, c, p in event_positions]
        emp_sum = 1.0

    mask_event = np.zeros_like(out, dtype=bool)
    for r, c, _ in event_positions:
        mask_event[r, c] = True

    q = np.zeros_like(out, dtype=float)
    for r, c, p in event_positions:
        q[r, c] = p

    rem_mass = max(0.0, 1.0 - emp_sum)
    non_base = out[~mask_event]
    non_sum = float(non_base.sum())
    if non_sum > 0:
        q[~mask_event] = non_base * (rem_mass / non_sum)
    elif (~mask_event).any():
        q[~mask_event] = rem_mass / float((~mask_event).sum())

    mixed = (1.0 - alpha) * out + alpha * q
    z = float(mixed.sum())
    if z <= 0:
        return out
    return mixed / z


def dixon_coles_tau(home_goals: int, away_goals: int, lam_home: float, lam_away: float, rho: float) -> float:
    """
    Dixon-Coles low-score interaction term.

    Applies only to scorelines:
      (0,0), (0,1), (1,0), (1,1)
    and returns 1.0 for all others.
    """
    x = int(home_goals)
    y = int(away_goals)
    lh = float(max(lam_home, 1e-12))
    la = float(max(lam_away, 1e-12))
    r = float(rho)

    if x == 0 and y == 0:
        return 1.0 - (lh * la * r)
    if x == 0 and y == 1:
        return 1.0 + (lh * r)
    if x == 1 and y == 0:
        return 1.0 + (la * r)
    if x == 1 and y == 1:
        return 1.0 - r
    return 1.0


def _apply_dixon_coles_adjustment(mat: np.ndarray, labels: list[str], lam_home: float, lam_away: float, rho: float) -> np.ndarray:
    """
    Applies the Dixon-Coles low-score adjustment then renormalizes.
    """
    adj = mat.copy()
    label_to_idx = {label: i for i, label in enumerate(labels)}

    for hs, as_ in ((0, 0), (0, 1), (1, 0), (1, 1)):
        hs_label = str(hs)
        as_label = str(as_)
        if hs_label in label_to_idx and as_label in label_to_idx:
            i = label_to_idx[hs_label]
            j = label_to_idx[as_label]
            tau = dixon_coles_tau(hs, as_, lam_home, lam_away, rho)
            if tau <= 0:
                tau = 1e-12
            adj[i, j] *= tau

    total = float(adj.sum())
    if total <= 0:
        return mat / float(max(mat.sum(), 1e-12))
    return adj / total


def load_artifact(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    try:
        artifact = joblib.load(path)
    except Exception as exc:
        runtime_sklearn = _get_runtime_sklearn_version()
        details = (
            f"Failed to load model artifact: {path}\n"
            f"Root error: {type(exc).__name__}: {exc}\n"
            f"Detected runtime scikit-learn: {runtime_sklearn or 'unknown'}\n"
            "This is commonly caused by loading an artifact created with a different "
            "scikit-learn version.\n"
            "Fix options:\n"
            "  1) Install the artifact-compatible version (recommended for this project: scikit-learn==1.6.1)\n"
            "  2) Retrain the model artifact in your current environment"
        )
        raise RuntimeError(details) from exc

    expected_sklearn = (
        artifact.get("training_environment", {}) if isinstance(artifact, dict) else {}
    ).get("scikit_learn_version")
    runtime_sklearn = _get_runtime_sklearn_version()
    if expected_sklearn and runtime_sklearn:
        if _major_minor(str(expected_sklearn)) != _major_minor(str(runtime_sklearn)):
            raise RuntimeError(
                "Artifact/runtime scikit-learn mismatch detected. "
                f"artifact={expected_sklearn}, runtime={runtime_sklearn}. "
                "Use the same major.minor version as the training environment or retrain the artifact."
            )

    return artifact


def _get_runtime_sklearn_version() -> str | None:
    try:
        return package_version("scikit-learn")
    except PackageNotFoundError:
        return None


def _major_minor(v: str) -> tuple[int, int] | None:
    parts = str(v).split(".")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def predict_expected_goals(artifact: Dict[str, Any], X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      (lambda_home, lambda_away) as numpy arrays.
    """
    model_home = artifact["model_home"]
    model_away = artifact["model_away"]
    lam_home = model_home.predict(X)
    lam_away = model_away.predict(X)

    # Guard against any numerical weirdness
    lam_home = np.clip(lam_home, 1e-6, None)
    lam_away = np.clip(lam_away, 1e-6, None)
    return lam_home, lam_away


def poisson_pmf_vector(lam: float, max_k: int) -> np.ndarray:
    """
    Numerically stable Poisson PMF vector using recursion:
      p0 = exp(-lam)
      p(k+1) = p(k) * lam/(k+1)

    Returns probabilities for k=0..max_k.
    """
    lam = float(max(lam, 1e-12))
    probs = np.zeros(max_k + 1, dtype=float)
    probs[0] = math.exp(-lam)
    for k in range(max_k):
        probs[k + 1] = probs[k] * lam / (k + 1)
    return probs


def scoreline_probability_matrix(
    lam_home: float,
    lam_away: float,
    max_goals: int = 6,
    include_tail_bucket: bool = True,
    rho: float | None = None,
) -> pd.DataFrame:
    """
    Build a (max_goals+1)x(max_goals+1) matrix of scoreline probabilities.
    If include_tail_bucket=True, adds a final 'max_goals+' row/column that
    contains probability mass for goals > max_goals.
    """
    if include_tail_bucket:
        ph = poisson_pmf_vector(lam_home, max_goals)
        pa = poisson_pmf_vector(lam_away, max_goals)
        tail_h = max(0.0, 1.0 - float(ph.sum()))
        tail_a = max(0.0, 1.0 - float(pa.sum()))
        ph = np.concatenate([ph, [tail_h]])
        pa = np.concatenate([pa, [tail_a]])

        labels = [str(i) for i in range(max_goals + 1)] + [f"{max_goals}+"]
        mat = np.outer(ph, pa)
        if rho is not None:
            mat = _apply_dixon_coles_adjustment(mat, labels, lam_home, lam_away, float(rho))
        return pd.DataFrame(mat, index=labels, columns=labels)

    # No tail bucket
    ph = poisson_pmf_vector(lam_home, max_goals)
    pa = poisson_pmf_vector(lam_away, max_goals)
    labels = [str(i) for i in range(max_goals + 1)]
    mat = np.outer(ph, pa)
    if rho is not None:
        mat = _apply_dixon_coles_adjustment(mat, labels, lam_home, lam_away, float(rho))
    else:
        mat = mat / mat.sum()
    return pd.DataFrame(mat, index=labels, columns=labels)


def top_scorelines(mat: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Returns a table of the top-N scorelines by probability.
    """
    s = mat.stack().reset_index()
    s.columns = ["home_goals", "away_goals", "prob"]
    s = s.sort_values("prob", ascending=False).head(top_n).reset_index(drop=True)
    s["score"] = s["home_goals"].astype(str) + "-" + s["away_goals"].astype(str)
    return s[["score", "prob", "home_goals", "away_goals"]]


def win_draw_loss_probs(lam_home: float, lam_away: float, max_k: int = 12, rho: float | None = None) -> Dict[str, float]:
    """
    Approximate W/D/L probabilities by enumerating goals 0..max_k and
    renormalizing (tail mass is typically tiny for soccer).

    Returns dict with keys: home_win, draw, away_win.
    """
    ph = poisson_pmf_vector(lam_home, max_k)
    pa = poisson_pmf_vector(lam_away, max_k)
    mat = np.outer(ph, pa)
    if rho is not None:
        labels = [str(i) for i in range(max_k + 1)]
        mat = _apply_dixon_coles_adjustment(mat, labels, lam_home, lam_away, float(rho))

    mass = float(mat.sum())
    if mass <= 0:
        return {"home_win": float("nan"), "draw": float("nan"), "away_win": float("nan")}

    mat = mat / mass
    home_win = float(np.tril(mat, k=-1).sum())  # i > j -> below diagonal? careful with orientation
    # Here rows=home goals, cols=away goals:
    # home_win: row > col => below diagonal (k=-1)
    home_win = float(np.tril(mat, k=-1).sum())
    draw = float(np.trace(mat))
    away_win = float(np.triu(mat, k=1).sum())
    return {"home_win": home_win, "draw": draw, "away_win": away_win}
