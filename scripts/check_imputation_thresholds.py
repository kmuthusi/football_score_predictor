"""CI helper: fail when imputation rates exceed configured thresholds.

Exits with status 1 when:
 - rows_with_any_imputation_prop > --max-row-prop, OR
 - any per_column_missing_rate for required model inputs > --max-col-rate

Returns 0 otherwise and prints a concise summary.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--report", type=str, default="imputation_report.json")
    p.add_argument("--max-row-prop", type=float, default=0.20,
                   help="Fail if proportion of rows with any imputation exceeds this.")
    p.add_argument("--max-col-rate", type=float, default=0.10,
                   help="Fail if any required column has missing rate above this.")
    args = p.parse_args()

    rpt_path = Path(args.report)
    if not rpt_path.exists():
        print(f"[IMPUTATION-CHECK] Report not found: {rpt_path}")
        sys.exit(1)

    data: Dict = json.loads(rpt_path.read_text(encoding="utf-8"))
    rows_prop = float(data.get("rows_with_any_imputation_prop", 0.0))
    col_rates = data.get("per_column_missing_rate", {}) or {}

    violations = []

    if rows_prop > float(args.max_row_prop):
        violations.append(f"rows_with_any_imputation_prop={rows_prop:.3f} > {args.max_row_prop:.3f}")

    high_cols = {c: r for c, r in col_rates.items() if float(r) > float(args.max_col_rate)}
    if high_cols:
        for c, r in sorted(high_cols.items(), key=lambda x: -float(x[1])):
            violations.append(f"col {c} missing_rate={float(r):.3f} > {args.max_col_rate:.3f}")

    if violations:
        print("[IMPUTATION-CHECK] FAILED: thresholds exceeded:")
        for v in violations:
            print("  -", v)
        print("[IMPUTATION-CHECK] See imputation_report.json for details.")
        sys.exit(1)

    print("[IMPUTATION-CHECK] OK — imputation rates within configured thresholds.")
    sys.exit(0)


if __name__ == "__main__":
    main()
