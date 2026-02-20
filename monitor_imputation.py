"""Compute imputation rates for model input columns and emit a JSON report.

This script is intended for CI/monitoring: run periodically (or on push) to detect
increases in feature missingness that may indicate data drift or upstream data
issues.

Outputs a JSON file (`imputation_report.json` by default) with per-column missing
rates and a short summary.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from features import (
    FeatureConfig,
    build_single_match_features,
    build_team_match_long,
    load_matches_csv,
    load_stadiums_csv,
)
from predict import load_artifact


def summarize_imputation(artifact_path: str, matches_path: str, stadiums_path: str | None, *,
                         recent_days: int | None = 365, last_n: int | None = 1000, out: str = "imputation_report.json") -> Dict:
    artifact = load_artifact(artifact_path)
    matches = load_matches_csv(matches_path)
    stadiums = load_stadiums_csv(stadiums_path) if stadiums_path else None

    cfg = FeatureConfig(**artifact.get("config", {}))
    req_cols: List[str] = list(artifact.get("cat_cols", []) + artifact.get("num_cols", []))

    # select candidate matches by recent_days or last_n
    frame = matches.sort_values("date").copy()
    if recent_days is not None:
        cutoff = pd.Timestamp(frame["date"].max()) - pd.Timedelta(days=int(recent_days))
        sel = frame[frame["date"] >= cutoff].copy()
    elif last_n is not None:
        sel = frame.tail(int(last_n)).copy()
    else:
        sel = frame.copy()

    n = len(sel)
    if n == 0:
        raise RuntimeError("No matches selected for imputation monitoring.")

    long_df = build_team_match_long(matches)

    missing_counts: Counter = Counter()
    rows_with_any = 0
    sample_imputed: List[Dict] = []

    for i, r in sel.iterrows():
        feat = build_single_match_features(
            div=r["div"],
            home_team=r["home_team"],
            away_team=r["away_team"],
            asof_date=pd.Timestamp(r["date"]),
            home_odds=r.get("home_odds"),
            draw_odds=r.get("draw_odds"),
            away_odds=r.get("away_odds"),
            long_df=long_df,
            stadiums=stadiums if stadiums is not None else pd.DataFrame(),
            config=cfg,
        )
        # ensure all required columns exist in the feature row; missing -> counts as imputed
        imputed_cols = []
        for c in req_cols:
            if c not in feat.columns or pd.isna(feat.at[0, c]):
                missing_counts[c] += 1
                imputed_cols.append(c)

        if imputed_cols:
            rows_with_any += 1
            if len(sample_imputed) < 10:
                sample_imputed.append({
                    "date": str(r["date"]),
                    "div": r["div"],
                    "home": r["home_team"],
                    "away": r["away_team"],
                    "imputed_cols": imputed_cols,
                })

    per_col_missing = {c: float(missing_counts.get(c, 0)) / float(max(1, n)) for c in req_cols}
    sorted_top = sorted(per_col_missing.items(), key=lambda x: x[1], reverse=True)

    output = {
        "artifact": str(artifact_path),
        "checked_matches": int(n),
        "period_start": str(sel["date"].min()) if not sel.empty else None,
        "period_end": str(sel["date"].max()) if not sel.empty else None,
        "rows_with_any_imputation": int(rows_with_any),
        "rows_with_any_imputation_prop": float(rows_with_any) / float(n),
        "per_column_missing_rate": per_col_missing,
        "top_imputed_columns": [{"col": k, "missing_rate": v} for k, v in sorted_top[:10]],
        "sample_imputed_matches": sample_imputed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    Path(out).write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor and report imputation rates for model inputs.")
    parser.add_argument("--artifact", type=str, required=True)
    parser.add_argument("--matches", type=str, required=True)
    parser.add_argument("--stadiums", type=str, default=None)
    parser.add_argument("--recent-days", type=int, default=365)
    parser.add_argument("--last-n", type=int, default=None)
    parser.add_argument("--out", type=str, default="imputation_report.json")
    parser.add_argument("--telemetry-url", type=str, default=None,
                        help="Optional URL to POST the generated report (best-effort).")
    args = parser.parse_args()

    report = summarize_imputation(
        artifact_path=args.artifact,
        matches_path=args.matches,
        stadiums_path=args.stadiums,
        recent_days=args.recent_days,
        last_n=args.last_n,
        out=args.out,
    )

    print(json.dumps(report, indent=2))

    # Optionally POST the report to telemetry (best-effort; do not raise on failure)
    turl = args.telemetry_url or None
    if turl:
        try:
            import requests

            with open(args.out, "rb") as fh:
                requests.post(turl, files={"report": fh}, timeout=3.0)
        except Exception:
            # Telemetry is optional — swallow errors
            pass


if __name__ == "__main__":
    main()
