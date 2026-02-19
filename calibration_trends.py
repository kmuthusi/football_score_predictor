from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_LOG_PATH = "models/low_score_calibration_log.csv"


def _fmt_delta(value: float | None, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:+.{digits}f}"


def print_trends(log_path: str | Path, last_n: int) -> None:
    path = Path(log_path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration log not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        print("Calibration log is empty.")
        return

    if "run_utc" in df.columns:
        df["run_utc"] = pd.to_datetime(df["run_utc"], errors="coerce", utc=True)
        df = df.sort_values("run_utc")

    n = int(max(1, last_n))
    window = df.tail(n).copy()

    latest = window.iloc[-1]
    prev = window.iloc[-2] if len(window) >= 2 else None
    first = window.iloc[0]

    print(f"Log file: {path}")
    print(f"Rows in log: {len(df)}")
    print(f"Window size: {len(window)} (requested {n})")
    print()

    cols_preview = [c for c in ["run_utc", "fit_dc", "dc_mode", "rho_global", "nll_independent", "nll_dc"] if c in window.columns]
    print("Recent runs:")
    print(window[cols_preview].to_string(index=False))
    print()

    print("Delta summary (latest vs previous run):")
    if prev is None:
        print("  Not enough rows in selected window to compute previous-run deltas.")
    else:
        for col in ["nll_independent", "nll_dc", "rho_global"]:
            if col in window.columns:
                delta = float(latest[col] - prev[col]) if pd.notna(latest[col]) and pd.notna(prev[col]) else None
                print(f"  {col}: {_fmt_delta(delta)}")
    print()
    print("Delta summary (latest vs first run in window):")
    for col in ["nll_independent", "nll_dc", "rho_global"]:
        if col in window.columns:
            delta = float(latest[col] - first[col]) if pd.notna(latest[col]) and pd.notna(first[col]) else None
            print(f"  {col}: {_fmt_delta(delta)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print trend deltas from low-score calibration log.")
    parser.add_argument("--log", type=str, default=DEFAULT_LOG_PATH, help="Path to low_score_calibration_log.csv")
    parser.add_argument("--last-n", type=int, default=10, help="Number of most-recent runs to inspect")
    args = parser.parse_args()
    print_trends(log_path=args.log, last_n=int(args.last_n))


if __name__ == "__main__":
    main()
