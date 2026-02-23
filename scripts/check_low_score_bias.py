"""Check low-score bias report and fail if biases exceed thresholds.

This is designed for CI; the JSON file is produced by ``low_score_analysis.py``.
"""
import argparse
import json
import sys


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--report", type=str, default="reports/low_score_bias.json")
    p.add_argument("--max-bias", type=float, default=0.02, help="max absolute bias allowed (fraction)")
    args = p.parse_args(argv)

    try:
        rpt = json.load(open(args.report))
    except Exception as e:
        print(f"[error] could not load report: {e}")
        return 2

    max_bias = float(args.max_bias)
    fail = False
    for key in ("bias_low1", "bias_low2"):
        val = float(rpt.get(key, 0.0))
        if abs(val) > max_bias:
            print(f"[check] {key}={val:.4f} exceeds threshold {max_bias:.4f}")
            fail = True
        else:
            print(f"[check] {key}={val:.4f} OK")

    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
