from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from config import DEFAULT_ARTIFACT_ABS_PATH, DEFAULT_MATCHES_REL_PATH, DEFAULT_STADIUMS_REL_PATH
from features import FeatureConfig, build_training_frame, load_matches_csv, load_stadiums_csv
from metrics import neg_log_likelihood, neg_log_likelihood_dixon_coles
from predict import calibrate_scoreline_matrix, load_artifact, scoreline_probability_matrix


def _score_to_label(v: int, max_goals: int) -> str:
    return str(int(v)) if int(v) <= int(max_goals) else f"{int(max_goals)}+"


def top1_ece(
    y_home: np.ndarray,
    y_away: np.ndarray,
    lam_home: np.ndarray,
    lam_away: np.ndarray,
    max_goals: int,
    rho: float | None,
    calibration_cfg: Dict[str, Any] | None,
    divs: np.ndarray,
    n_bins: int = 10,
) -> float:
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
        mat = calibrate_scoreline_matrix(mat, calibration_cfg, div=str(divs[i]))
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

    if len(conf) == 0:
        return float("nan")

    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
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
        ece += (n_b / max(len(conf), 1)) * abs(acc_b - conf_b)

    return float(ece)


def exact_score_top1_accuracy(
    y_home: np.ndarray,
    y_away: np.ndarray,
    lam_home: np.ndarray,
    lam_away: np.ndarray,
    max_goals: int,
    rho: float | None,
    calibration_cfg: Dict[str, Any] | None,
    divs: np.ndarray,
) -> float:
    correct = 0
    n = len(y_home)
    for i in range(n):
        mat = scoreline_probability_matrix(
            float(lam_home[i]),
            float(lam_away[i]),
            max_goals=int(max_goals),
            include_tail_bucket=True,
            rho=float(rho) if rho is not None else None,
        )
        mat = calibrate_scoreline_matrix(mat, calibration_cfg, div=str(divs[i]))
        arr = mat.to_numpy(dtype=float)
        idx = int(np.argmax(arr))
        r, c = divmod(idx, arr.shape[1])
        pred_h = str(mat.index[r])
        pred_a = str(mat.columns[c])
        true_h = _score_to_label(int(y_home[i]), int(max_goals))
        true_a = _score_to_label(int(y_away[i]), int(max_goals))
        correct += int(pred_h == true_h and pred_a == true_a)
    return float(correct / max(n, 1))


def resolve_split(frame: pd.DataFrame, artifact: Dict[str, Any], test_days: int | None) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if artifact.get("cutoff_date"):
        cutoff = pd.Timestamp(str(artifact["cutoff_date"]))
        train_df = frame[frame["date"] < cutoff].copy()
        test_df = frame[frame["date"] >= cutoff].copy()
        return train_df, test_df, str(cutoff.date())

    if test_days is None:
        test_days = 365
    max_date = pd.Timestamp(frame["date"].max())
    cutoff = max_date - pd.Timedelta(days=int(test_days))
    train_df = frame[frame["date"] < cutoff].copy()
    test_df = frame[frame["date"] >= cutoff].copy()
    return train_df, test_df, str(cutoff.date())


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible held-out evaluation for scoreline model artifacts.")
    parser.add_argument("--artifact", type=str, default=DEFAULT_ARTIFACT_ABS_PATH)
    parser.add_argument("--matches", type=str, default=DEFAULT_MATCHES_REL_PATH)
    parser.add_argument("--stadiums", type=str, default=DEFAULT_STADIUMS_REL_PATH)
    parser.add_argument("--test-days", type=int, default=None, help="Fallback only when artifact has no cutoff_date.")
    parser.add_argument("--bins", type=int, default=10, help="Number of bins for top-1 ECE.")
    parser.add_argument("--json", action="store_true", help="Emit JSON output only.")
    args = parser.parse_args()

    artifact_path = Path(args.artifact)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")

    artifact = load_artifact(artifact_path)
    cfg_raw = artifact.get("config", {})
    config = FeatureConfig(
        windows=tuple(cfg_raw.get("windows", [5, 10])),
        max_goals=int(cfg_raw.get("max_goals", 6)),
        use_travel_distance=bool(cfg_raw.get("use_travel_distance", True)),
        ewm_span=int(cfg_raw.get("ewm_span", 9)),
        use_ewm_features=bool(cfg_raw.get("use_ewm_features", True)),
        use_adjusted_features=bool(cfg_raw.get("use_adjusted_features", True)),
    )

    matches = load_matches_csv(args.matches)
    stadiums = load_stadiums_csv(args.stadiums) if args.stadiums else None
    frame = build_training_frame(matches, stadiums, config)
    frame = frame.dropna(subset=["home_goals", "away_goals"]).copy()

    train_df, test_df, cutoff_date = resolve_split(frame, artifact, args.test_days)
    if len(test_df) == 0:
        raise RuntimeError("Resolved test split is empty; cannot evaluate.")

    cat_cols = list(artifact.get("cat_cols", []))
    num_cols = list(artifact.get("num_cols", []))
    missing = [c for c in cat_cols + num_cols if c not in test_df.columns]
    if missing:
        raise ValueError(f"Missing required model feature columns in evaluation frame: {missing}")

    X_test = test_df[cat_cols + num_cols]
    y_home_test = test_df["home_goals"].astype(int).to_numpy()
    y_away_test = test_df["away_goals"].astype(int).to_numpy()
    div_test = test_df["div"].astype(str).to_numpy()

    model_home = artifact["model_home"]
    model_away = artifact["model_away"]
    lam_home = np.clip(model_home.predict(X_test), 1e-6, None)
    lam_away = np.clip(model_away.predict(X_test), 1e-6, None)

    rho = artifact.get("dixon_coles", {}).get("rho_global", artifact.get("dixon_coles_rho"))
    rho = float(rho) if rho is not None else None
    calibration_cfg = artifact.get("scoreline_calibration", None)
    rolling_bt = artifact.get("diagnostics", {}).get("rolling_backtest", {})
    bt_enabled = bool(rolling_bt.get("enabled", False))
    bt_n_folds = int(rolling_bt.get("n_folds", 0) or 0)
    bt_nll_mean = rolling_bt.get("nll_mean", None)
    bt_nll_std = rolling_bt.get("nll_std", None)

    if bt_nll_mean is not None:
        bt_nll_mean = float(bt_nll_mean)
    if bt_nll_std is not None:
        bt_nll_std = float(bt_nll_std)

    ind_nll = float(neg_log_likelihood(y_home_test, y_away_test, lam_home, lam_away))
    dc_nll = (
        float(neg_log_likelihood_dixon_coles(y_home_test, y_away_test, lam_home, lam_away, rho=rho))
        if rho is not None
        else None
    )

    ece_ind = float(
        top1_ece(
            y_home=y_home_test,
            y_away=y_away_test,
            lam_home=lam_home,
            lam_away=lam_away,
            max_goals=int(config.max_goals),
            rho=None,
            calibration_cfg=None,
            divs=div_test,
            n_bins=int(args.bins),
        )
    )
    ece_dc = (
        float(
            top1_ece(
                y_home=y_home_test,
                y_away=y_away_test,
                lam_home=lam_home,
                lam_away=lam_away,
                max_goals=int(config.max_goals),
                rho=rho,
                calibration_cfg=None,
                divs=div_test,
                n_bins=int(args.bins),
            )
        )
        if rho is not None
        else None
    )
    ece_dc_cal = (
        float(
            top1_ece(
                y_home=y_home_test,
                y_away=y_away_test,
                lam_home=lam_home,
                lam_away=lam_away,
                max_goals=int(config.max_goals),
                rho=rho,
                calibration_cfg=calibration_cfg,
                divs=div_test,
                n_bins=int(args.bins),
            )
        )
        if rho is not None
        else None
    )

    exact_top1_ind = float(
        exact_score_top1_accuracy(
            y_home=y_home_test,
            y_away=y_away_test,
            lam_home=lam_home,
            lam_away=lam_away,
            max_goals=int(config.max_goals),
            rho=None,
            calibration_cfg=None,
            divs=div_test,
        )
    )
    exact_top1_dc_cal = float(
        exact_score_top1_accuracy(
            y_home=y_home_test,
            y_away=y_away_test,
            lam_home=lam_home,
            lam_away=lam_away,
            max_goals=int(config.max_goals),
            rho=rho,
            calibration_cfg=calibration_cfg,
            divs=div_test,
        )
    )

    output = {
        "artifact": str(artifact_path),
        "matches": str(args.matches),
        "cutoff_date": cutoff_date,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "max_goals": int(config.max_goals),
        "rho_global": rho,
        "metrics": {
            "ind_nll": ind_nll,
            "dc_nll": dc_nll,
            "top1_ece_ind": ece_ind,
            "top1_ece_dc": ece_dc,
            "top1_ece_dc_calibrated": ece_dc_cal,
            "exact_top1_ind": exact_top1_ind,
            "exact_top1_dc_calibrated": exact_top1_dc_cal,
            "rolling_backtest_enabled": bt_enabled,
            "rolling_backtest_n_folds": bt_n_folds,
            "rolling_backtest_nll_mean": bt_nll_mean,
            "rolling_backtest_nll_std": bt_nll_std,
        },
    }

    if args.json:
        print(json.dumps(output, indent=2))
        return

    print("=== Reproducible Evaluation ===")
    print(f"Artifact:   {output['artifact']}")
    print(f"Matches:    {output['matches']}")
    print(f"Cutoff:     {output['cutoff_date']}")
    print(f"Rows:       train={output['train_rows']:,}, test={output['test_rows']:,}")
    print(f"rho_global: {output['rho_global']}")
    print()
    print("Metrics:")
    print(f"  ind_nll:                 {output['metrics']['ind_nll']:.6f}")
    print(f"  dc_nll:                  {output['metrics']['dc_nll'] if output['metrics']['dc_nll'] is not None else 'n/a'}")
    print(f"  top1_ece_ind:            {output['metrics']['top1_ece_ind']:.6f}")
    print(f"  top1_ece_dc:             {output['metrics']['top1_ece_dc'] if output['metrics']['top1_ece_dc'] is not None else 'n/a'}")
    print(
        "  top1_ece_dc_calibrated: "
        f"{output['metrics']['top1_ece_dc_calibrated'] if output['metrics']['top1_ece_dc_calibrated'] is not None else 'n/a'}"
    )
    print(f"  exact_top1_ind:          {output['metrics']['exact_top1_ind']:.6f}")
    print(f"  exact_top1_dc_calibrated:{output['metrics']['exact_top1_dc_calibrated']:.6f}")
    print(
        "  rolling_backtest:        "
        f"enabled={output['metrics']['rolling_backtest_enabled']}, "
        f"folds={output['metrics']['rolling_backtest_n_folds']}, "
        f"mean_nll={output['metrics']['rolling_backtest_nll_mean']}, "
        f"std_nll={output['metrics']['rolling_backtest_nll_std']}"
    )


if __name__ == "__main__":
    main()
