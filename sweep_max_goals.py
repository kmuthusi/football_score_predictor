from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), check=True, text=True, capture_output=True)


def rank(values: list[float], reverse: bool = False) -> list[int]:
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=reverse)
    out = [0] * len(values)
    for pos, idx in enumerate(order, start=1):
        out[idx] = pos
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep max_goals and select best held-out setting.")
    parser.add_argument("--matches", type=str, default="data/spi_matches.csv")
    parser.add_argument("--stadiums", type=str, default="data/stadium_coordinates_completed_full.csv")
    parser.add_argument("--base-out", type=str, default="models/sweeps/max_goals")
    parser.add_argument("--candidates", type=int, nargs="+", default=[4, 5, 6, 7, 8])
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--backtest-folds", type=int, default=3,
                        help="Rolling-origin folds passed to train.py for diagnostics and selection stability.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    base_out = root / args.base_out
    base_out.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    rows: list[dict[str, float | int | str]] = []

    for mg in args.candidates:
        out_dir = base_out / f"mg_{int(mg)}"
        out_dir.mkdir(parents=True, exist_ok=True)

        train_cmd = [
            py,
            "train.py",
            "--matches",
            str(args.matches),
            "--stadiums",
            str(args.stadiums),
            "--out",
            str(out_dir),
            "--max-goals",
            str(int(mg)),
            "--max-iter",
            str(int(args.max_iter)),
            "--backtest-folds",
            str(int(args.backtest_folds)),
        ]
        print(f"[Sweep] Training max_goals={mg} ...")
        run_command(train_cmd, cwd=root)

        eval_cmd = [
            py,
            "evaluate_reproducible.py",
            "--artifact",
            str(out_dir / "score_models.joblib"),
            "--matches",
            str(args.matches),
            "--stadiums",
            str(args.stadiums),
            "--json",
        ]
        print(f"[Sweep] Evaluating max_goals={mg} ...")
        proc = run_command(eval_cmd, cwd=root)
        payload = json.loads(proc.stdout)
        metrics = payload["metrics"]

        top1_ece = metrics.get("top1_ece_dc_calibrated")
        if top1_ece is None:
            top1_ece = metrics.get("top1_ece_ind")

        exact_top1 = metrics.get("exact_top1_dc_calibrated")
        if exact_top1 is None:
            exact_top1 = metrics.get("exact_top1_ind")

        rows.append(
            {
                "max_goals": int(mg),
                "artifact": str(out_dir / "score_models.joblib"),
                "ind_nll": float(metrics["ind_nll"]),
                "top1_ece": float(top1_ece),
                "exact_top1": float(exact_top1),
                "rolling_backtest_nll_mean": (
                    float(metrics["rolling_backtest_nll_mean"])
                    if metrics.get("rolling_backtest_nll_mean") is not None
                    else None
                ),
                "rolling_backtest_n_folds": int(metrics.get("rolling_backtest_n_folds", 0) or 0),
            }
        )

    ind_ranks = rank([float(r["ind_nll"]) for r in rows], reverse=False)
    ece_ranks = rank([float(r["top1_ece"]) for r in rows], reverse=False)
    acc_ranks = rank([float(r["exact_top1"]) for r in rows], reverse=True)
    has_bt_metric = all(
        (r.get("rolling_backtest_nll_mean") is not None) and (int(r.get("rolling_backtest_n_folds", 0)) > 0)
        for r in rows
    )
    if has_bt_metric:
        bt_ranks = rank([float(r["rolling_backtest_nll_mean"]) for r in rows], reverse=False)
    else:
        bt_ranks = None

    for i, row in enumerate(rows):
        row["rank_ind_nll"] = ind_ranks[i]
        row["rank_top1_ece"] = ece_ranks[i]
        row["rank_exact_top1"] = acc_ranks[i]
        if bt_ranks is not None:
            row["rank_rolling_backtest_nll"] = bt_ranks[i]
            row["rank_sum"] = int(ind_ranks[i] + ece_ranks[i] + acc_ranks[i] + bt_ranks[i])
        else:
            row["rank_rolling_backtest_nll"] = None
            row["rank_sum"] = int(ind_ranks[i] + ece_ranks[i] + acc_ranks[i])

    rows_sorted = sorted(rows, key=lambda r: (int(r["rank_sum"]), -float(r["exact_top1"])))
    best = rows_sorted[0]

    print("\n=== Sweep Results (lower rank_sum is better) ===")
    for row in rows_sorted:
        print(
            "max_goals={max_goals} | ind_nll={ind_nll:.6f} | top1_ece={top1_ece:.6f} | "
            "exact_top1={exact_top1:.6f} | rank_sum={rank_sum}".format(**row)
        )

    output = {
        "candidates": [int(x) for x in args.candidates],
        "selection_rule": (
            "minimize rank_sum of (ind_nll asc, top1_ece asc, exact_top1 desc, rolling_backtest_nll_mean asc)"
            if bt_ranks is not None
            else "minimize rank_sum of (ind_nll asc, top1_ece asc, exact_top1 desc)"
        ),
        "backtest_folds": int(args.backtest_folds),
        "used_rolling_backtest_in_rank": bool(bt_ranks is not None),
        "best": best,
        "results": rows_sorted,
    }

    out_json = base_out / "sweep_results.json"
    out_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nBest max_goals: {best['max_goals']}")
    print(f"Saved sweep summary: {out_json}")


if __name__ == "__main__":
    main()
