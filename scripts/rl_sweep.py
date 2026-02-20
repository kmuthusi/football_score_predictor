"""Grid search over (bet_penalty, ev_threshold) using 1-epoch smoke training.

Usage:
  python scripts/rl_sweep.py

Outputs a JSON table of results sorted by ROI (on staked).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any

import argparse
import csv

DEFAULT_BET_PENALTIES = [0.0, 0.0005, 0.001, 0.0025, 0.005, 0.01]
DEFAULT_EV_THRESHOLDS = [0.0, 0.005, 0.01, 0.02, 0.05]

ARTIFACT = "models/score_models.joblib"
MATCHES = "data/spi_matches.csv"
STADIUMS = "data/stadium_coordinates.csv"


def run_cmd(cmd: List[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.stdout + "\n" + p.stderr


def parse_rl_eval(output: str) -> Dict[str, Any]:
    import re
    out = {}
    m = re.search(r"Bets placed:\s*([0-9,]+)", output)
    if m:
        out["bets_placed"] = int(m.group(1).replace(",", ""))
    m = re.search(r"ROI \(on staked\):\s*([\-0-9\.]+)%", output)
    if m:
        out["roi_on_staked_pct"] = float(m.group(1))
    m = re.search(r"Final bankroll:\s*([0-9\.,]+)", output)
    if m:
        out["final_bankroll"] = float(m.group(1).replace(",", ""))
    return out


def policy_name(bp: float, ev: float) -> str:
    return f"models/rl_policy_bp{int(bp*1000)}_ev{int(ev*1000)}.joblib"


def parse_list(s: str, default: List[float]) -> List[float]:
    if not s:
        return default
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [float(x) for x in parts]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bet-penalties", type=str, default="", help="Comma-separated bet_penalty values")
    parser.add_argument("--ev-thresholds", type=str, default="", help="Comma-separated ev_threshold values")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--out-json", type=str, default="reports/rl_sweep.json")
    parser.add_argument("--out-csv", type=str, default="reports/rl_sweep.csv")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output JSON (skip completed combos)")
    args = parser.parse_args()

    bet_penalties = parse_list(args.bet_penalties, DEFAULT_BET_PENALTIES)
    ev_thresholds = parse_list(args.ev_thresholds, DEFAULT_EV_THRESHOLDS)

    Path("reports").mkdir(parents=True, exist_ok=True)

    # support resume by loading existing results
    results_map = {}
    results: List[Dict[str, Any]] = []
    if args.resume and Path(args.out_json).exists():
        try:
            with open(args.out_json, "r", encoding="utf-8") as fj:
                existing = json.load(fj)
                for r in existing:
                    key = (float(r.get("bet_penalty", 0.0)), float(r.get("ev_threshold", 0.0)))
                    results_map[key] = r
                    results.append(r)
            print(f"Resuming sweep: loaded {len(results)} completed entries from {args.out_json}")
        except Exception:
            print("Warning: failed to load existing JSON for resume; starting fresh.")

    total = len(bet_penalties) * len(ev_thresholds)
    done = len(results_map)
    print(f"Sweep grid: {total} combinations ({done} already done) — starting.")

    for bp in bet_penalties:
        for ev in ev_thresholds:
            key = (float(bp), float(ev))
            if key in results_map:
                print(f"Skipping already-done combo bet_penalty={bp}, ev_threshold={ev}")
                continue

            out_policy = policy_name(bp, ev)
            print(f"Running sweep: bet_penalty={bp}, ev_threshold={ev} -> {out_policy}")

            # train
            cmd_train = [
                "python",
                "rl_train.py",
                "--artifact",
                ARTIFACT,
                "--matches",
                MATCHES,
                "--stadiums",
                STADIUMS,
                "--epochs",
                str(args.epochs),
                "--save",
                out_policy,
                "--no-test-eval",
                "--bet-penalty",
                str(bp),
                "--ev-threshold",
                str(ev),
            ]
            _ = run_cmd(cmd_train)

            # eval
            cmd_eval = [
                "python",
                "rl_eval.py",
                "--policy",
                out_policy,
                "--artifact",
                ARTIFACT,
                "--matches",
                MATCHES,
                "--stadiums",
                STADIUMS,
                "--initial-bankroll",
                "1000",
                "--stake-frac",
                "0.01",
                "--max-stake-frac",
                "0.05",
                "--test-days",
                "365",
                "--ev-threshold",
                str(ev),
            ]
            out_eval = run_cmd(cmd_eval)
            parsed = parse_rl_eval(out_eval)
            parsed.update({"bet_penalty": bp, "ev_threshold": ev, "policy": out_policy})
            results.append(parsed)

            # write incremental outputs after each combo to survive interruptions
            results_sorted = sorted(results, key=lambda x: (x.get("roi_on_staked_pct") if x.get("roi_on_staked_pct") is not None else -9999), reverse=True)
            with open(args.out_json, "w", encoding="utf-8") as fj:
                json.dump(results_sorted, fj, indent=2)
            with open(args.out_csv, "w", newline="", encoding="utf-8") as fc:
                writer = csv.DictWriter(fc, fieldnames=["bet_penalty", "ev_threshold", "bets_placed", "roi_on_staked_pct", "final_bankroll", "policy"])
                writer.writeheader()
                for r in results_sorted:
                    writer.writerow({k: r.get(k, "") for k in writer.fieldnames})

    print(f"Completed sweep. Saved JSON -> {args.out_json}")
    print(f"Completed sweep. Saved CSV  -> {args.out_csv}")


if __name__ == "__main__":
    main()
