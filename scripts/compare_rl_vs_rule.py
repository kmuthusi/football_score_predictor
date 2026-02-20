"""Compare an RL policy's backtest to simple EV rule baselines.

Prints a short JSON summary with:
 - rl: bets placed, ROI (on staked), final bankroll
 - rule: simulated ROI_on_staked for several EV thresholds (from ev_diagnostic)

Usage:
  python scripts/compare_rl_vs_rule.py --policy models/rl_policy.joblib --artifact models/score_models.joblib
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

# ensure project root is on sys.path when executed as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ev_diagnostic import summarize_ev_distribution


def parse_rl_eval_output(text: str) -> Dict[str, Any]:
    out = {}
    m = re.search(r"Bets placed:\s*([0-9,]+)", text)
    if m:
        out["bets_placed"] = int(m.group(1).replace(",", ""))
    m = re.search(r"ROI \(on staked\):\s*([\-0-9\.]+)%", text)
    if m:
        out["roi_on_staked_pct"] = float(m.group(1))
    m = re.search(r"Final bankroll:\s*([0-9\.,]+)", text)
    if m:
        out["final_bankroll"] = float(m.group(1).replace(",", ""))
    return out


def run_rl_eval(policy: str, artifact: str, matches: str, stadiums: str | None) -> Dict[str, Any]:
    cmd = ["python", "rl_eval.py", "--policy", policy, "--artifact", artifact, "--matches", matches]
    if stadiums:
        cmd += ["--stadiums", stadiums]
    p = subprocess.run(cmd, capture_output=True, text=True)
    stdout = p.stdout + "\n" + p.stderr
    return parse_rl_eval_output(stdout)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--artifact", required=True)
    p.add_argument("--matches", default="data/spi_matches.csv")
    p.add_argument("--stadiums", default=None)
    p.add_argument("--recent-days", type=int, default=365)
    args = p.parse_args()

    rule = summarize_ev_distribution(args.artifact, args.matches, args.stadiums, recent_days=args.recent_days)
    rl = run_rl_eval(args.policy, args.artifact, args.matches, args.stadiums)

    summary = {"rule_baseline": rule, "rl_policy": rl}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
