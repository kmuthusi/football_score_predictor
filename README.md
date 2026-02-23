# Football Prediction App

[![CI — main](https://github.com/kmuthusi/football_score_predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/kmuthusi/football_score_predictor/actions)

Predict exact football scoreline probabilities from historical match data — train models, evaluate held‑out performance, and run a Streamlit UI for interactive exploration.

For full technical and operational documentation, see `PROJECT_DOCUMENTATION.md`.

---

## Quickstart — 3 steps

1) Install dependencies

```bash
pip install -r requirements.txt
```

2) Train a model artifact (creates `models/score_models.joblib`)

```bash
python train.py --matches data/spi_matches.csv --stadiums data/stadium_coordinates.csv --out models --max-iter 1000
```

3) Launch the Streamlit app

```bash
streamlit run app.py
```

---

## Common commands

- Run the test suite

```bash
python -m unittest discover -s tests -p "test_*.py"
```

- Reproducible held-out evaluation (CLI / JSON)

```bash
python evaluate_reproducible.py --artifact models/score_models.joblib --matches data/spi_matches.csv --stadiums data/stadium_coordinates.csv
python evaluate_reproducible.py --artifact models/score_models.joblib --matches data/spi_matches.csv --stadiums data/stadium_coordinates.csv --json
```

- View recent calibration trends

```bash
python -m calibration_trends --last-n 10
```

- Recommended (enhanced) training run

```bash
python train.py --matches data/spi_matches.csv --stadiums data/stadium_coordinates.csv --out models \
  --fit-dc --tune-decay --tune-metric dc_nll --dc-optimize-oot \
  --decay-candidates 0 365 730 --val-days 180 --max-iter 1000 \
  --use-ewm-features --use-adjusted-features
```

- Optional: low-score mixture extension

```bash
python train.py --matches data/spi_matches.csv --stadiums data/stadium_coordinates.csv --out models \
  --fit-dc --fit-score-calibration --fit-low-score-mixture --max-iter 1000
```

### RL quick commands (optional)

Train a policy (example):

```bash
python rl_train.py --artifact models/score_models.joblib --matches data/spi_matches.csv \
  --stadiums data/stadium_coordinates.csv --epochs 30 \
  --bet-penalty 0.01 --low-score-penalty 0.05
```

Backtest / evaluate a saved policy:

```bash
python rl_eval.py --policy models/rl_policy.joblib --artifact models/score_models.joblib \
  --matches data/spi_matches.csv --stadiums data/stadium_coordinates.csv
```

Add `--low-score-penalty` and `--bet-penalty` to training to discourage betting on low-scoring games and over-trading.  Run `scripts/check_rl_policy_safety.py` on any policy before deploying it.

---

## Low-score bias and CI checks

A diagnostic script writes `reports/low_score_bias.json` comparing empirical low-score rates to model predictions.  CI runs:

```bash
python scripts/low_score_analysis.py
python scripts/check_low_score_bias.py --max-bias 0.02
```

Failures indicate calibration drift on low‑scoring games.


## CI monitoring & alerts

We automatically monitor imputation (missingness) in model inputs and alert when thresholds are exceeded.

What happens in CI

- `monitor_imputation.py` produces `imputation_report.json` (per-column missing rates + sample matches).
- `scripts/check_imputation_thresholds.py` will fail the job when configured thresholds are exceeded.
- On failure the workflow may open a GitHub **Issue**, send a Slack alert (if `SLACK_WEBHOOK_URL` is set), and POST the report to `IMPUTATION_TELEMETRY_URL`.

Default thresholds (adjustable in `.github/workflows/ci.yml`):
- Rows with any imputation &lt; 0.10 (10%)
- Per-column missing rate &lt; 0.05 (5%)

Run locally

```bash
python monitor_imputation.py --artifact models/score_models.joblib --matches data/spi_matches.csv --stadiums data/stadium_coordinates.csv --recent-days 365 --out imputation_report.json
python scripts/check_imputation_thresholds.py --report imputation_report.json --max-row-prop 0.10 --max-col-rate 0.05
```

---

## Troubleshooting & notes

- If `app.py` cannot find the artifact, retrain and confirm `models/score_models.joblib` exists.
- If training convergence is weak, increase `--max-iter`.

### Artifact compatibility

If loading `models/score_models.joblib` fails with sklearn-related unpickle errors, ensure your runtime `scikit-learn` matches the training environment (recommended: `scikit-learn==1.6.1`).

```bash
pip install scikit-learn==1.6.1
```

Or retrain in your environment:

```bash
python train.py --matches data/spi_matches.csv --stadiums data/stadium_coordinates.csv --out models --max-iter 1000
```

---

## Development

- Run unit tests: `python -m unittest discover -s tests -p "test_*.py"`
- CI workflows: see `.github/workflows/ci.yml` (includes imputation monitoring and an `rl-smoke` job)

For more details and design notes, see `PROJECT_DOCUMENTATION.md`.

