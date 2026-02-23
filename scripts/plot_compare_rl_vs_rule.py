"""Run RL evaluation and rule baseline comparison, then save a comparison plot.

Produces:
 - reports/rl_vs_rule_summary.json  (raw numeric summary)
 - reports/rl_vs_rule_comparison.png (visual comparison: rule ROI vs RL)

Usage:
  python scripts/plot_compare_rl_vs_rule.py --policy models/rl_policy.joblib --artifact models/score_models.joblib
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

# Ensure package import from repo root when running as script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.compare_rl_vs_rule import run_rl_eval
from scripts.ev_diagnostic import summarize_ev_distribution

import matplotlib.pyplot as plt
import numpy as np
import argparse


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--artifact", required=True)
    p.add_argument("--matches", default="data/spi_matches.csv")
    p.add_argument("--stadiums", default=None)
    p.add_argument("--recent-days", type=int, default=365)
    args = p.parse_args()

    # Gather data
    rule = summarize_ev_distribution(args.artifact, args.matches, args.stadiums, recent_days=args.recent_days)
    rl = run_rl_eval(args.policy, args.artifact, args.matches, args.stadiums)

    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {"rule_baseline": rule, "rl_policy": rl}
    json_path = out_dir / "rl_vs_rule_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Prepare plot: rule thresholds (x) vs ROI_on_staked (%) and horizontal RL ROI line
    thresholds = [r["threshold"] for r in rule.get("rule_sim", [])]
    rule_roi_pct = [100.0 * float(r.get("roi_on_staked", 0.0)) for r in rule.get("rule_sim", [])]

    rl_roi_pct = None
    if isinstance(rl.get("roi_on_staked_pct"), (int, float)):
        rl_roi_pct = float(rl.get("roi_on_staked_pct"))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, rule_roi_pct, marker="o", linestyle="-", label="Rule baseline (ROI on staked)")
    ax.set_xlabel("EV threshold (p*)")
    ax.set_ylabel("ROI on staked (%)")
    ax.set_title("RL policy vs EV‑rule baseline: ROI on staked")
    ax.grid(axis="y", alpha=0.3)

    if rl_roi_pct is not None:
        ax.axhline(rl_roi_pct, color="C3", linestyle="--", label=f"RL policy ROI: {rl_roi_pct:.2f}%")
        # annotate
        bets = rl.get("bets_placed")
        fb = rl.get("final_bankroll")
        ax.text(0.95, 0.05, f"RL: bets={bets}\nbankroll={fb}", transform=ax.transAxes, ha="right", va="bottom", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.legend()
    plt.tight_layout()

    out_png = out_dir / "rl_vs_rule_comparison.png"
    fig.savefig(out_png, dpi=150)

    print(f"Wrote: {json_path}\nWrote: {out_png}")


if __name__ == "__main__":
    main()
