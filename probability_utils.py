from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
import pandas as pd


def wdl_from_scoreline_matrix(mat: pd.DataFrame) -> Tuple[float, float, float]:
    arr = mat.to_numpy(dtype=float)
    p_home = float(np.tril(arr, k=-1).sum())
    p_draw = float(np.trace(arr))
    p_away = float(np.triu(arr, k=1).sum())
    return p_home, p_draw, p_away


def implied_probs_from_odds(
    home_odds: float,
    draw_odds: float,
    away_odds: float,
    *,
    invalid_fallback: Literal["zeros", "uniform"] = "zeros",
) -> Tuple[float, float, float]:
    ho = float(home_odds)
    do = float(draw_odds)
    ao = float(away_odds)
    qh = 1.0 / max(ho, 1e-12)
    qd = 1.0 / max(do, 1e-12)
    qa = 1.0 / max(ao, 1e-12)
    s = qh + qd + qa
    if (not np.isfinite(s)) or s <= 0:
        if invalid_fallback == "uniform":
            return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        return (0.0, 0.0, 0.0)
    return qh / s, qd / s, qa / s
