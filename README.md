# Football Prediction App

[![CI — main](https://github.com/kmuthusi/football_score_predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/kmuthusi/football_score_predictor/actions)

Predict exact football scoreline probabilities from historical match data.

For full technical and operational documentation, see:

- [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)

## Quickstart

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Train a model artifact

```bash
python train.py --matches data/spi_matches.csv --stadiums data/stadium_coordinates.csv --out models --max-iter 1000
```

This creates:

- `models/score_models.joblib`

### 3) Launch the Streamlit app

```bash
streamlit run app.py
```

## Common Commands

### Enhanced training run (recommended)

```bash
python train.py --matches data/spi_matches.csv --stadiums data/stadium_coordinates.csv --out models --fit-dc --tune-decay --tune-metric dc_nll --dc-optimize-oot --decay-candidates 0 365 730 --val-days 180 --max-iter 1000 --use-ewm-features --use-adjusted-features
```

Optional low-score mixture extension (only if held-out diagnostics still underfit low scores):

```bash
python train.py --matches data/spi_matches.csv --stadiums data/stadium_coordinates.csv --out models --fit-dc --fit-score-calibration --fit-low-score-mixture --max-iter 1000
```

### View calibration trends

```bash
python -m calibration_trends --last-n 10
```

### Run tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

### Reproducible held-out evaluation

```bash
python evaluate_reproducible.py --artifact models/score_models.joblib --matches data/spi_matches.csv --stadiums data/stadium_coordinates.csv
```

JSON output for logging/CI:

```bash
python evaluate_reproducible.py --artifact models/score_models.joblib --matches data/spi_matches.csv --stadiums data/stadium_coordinates.csv --json
```

### RL quick commands (optional)

Train (30 epochs):

```bash
python rl_train.py --artifact models/score_models.joblib --matches data/spi_matches.csv --stadiums data/stadium_coordinates.csv --epochs 30
```

Evaluate / backtest a saved policy:

```bash
python rl_eval.py --policy models/rl_policy.joblib --artifact models/score_models.joblib --matches data/spi_matches.csv --stadiums data/stadium_coordinates.csv
```

## Reinforcement-learning policy (optional)

A simple REINFORCE policy trainer / evaluator is included for experimentation. Artifacts are saved to `models/rl_policy.joblib` by default.

Train a quick policy (smoke run):

```bash
python rl_train.py --artifact models/score_models.joblib --matches data/spi_matches.csv --stadiums data/stadium_coordinates.csv --epochs 30 --save models/rl_policy.joblib
```

Run a greedy backtest of a saved policy:

```bash
python rl_eval.py --policy models/rl_policy.joblib --artifact models/score_models.joblib --matches data/spi_matches.csv --stadiums data/stadium_coordinates.csv
```

CI integration: a lightweight smoke job (`rl-smoke`) trains for 1 epoch and runs the backtest on PRs (see `.github/workflows/ci.yml`). The CI job is optimized to run only when RL-related code or feature/predict logic changes and will reuse a cached model artifact when available to keep PR runs fast.

## Notes

- If the app cannot find the artifact, retrain and confirm `models/score_models.joblib` exists.
- If convergence is weak, increase `--max-iter`.
- Odds inputs are optional but typically improve signal quality.

### Artifact compatibility (troubleshooting)

If Streamlit fails while loading `models/score_models.joblib` (for example with sklearn unpickle errors such as `_RemainderColsList`), your runtime sklearn version likely differs from the version used to train/save the artifact.

Fix options:

1. Install the compatible version and rerun:

```bash
pip install scikit-learn==1.6.1
```

2. Or retrain in your current environment to regenerate the artifact:

```bash
python train.py --matches data/spi_matches.csv --stadiums data/stadium_coordinates.csv --out models --max-iter 1000
```

## CI monitoring & alerts ⚠️

The repository includes automated imputation monitoring and alerting to catch upstream data issues that increase missingness in model inputs.

What the CI does

- Runs `monitor_imputation.py` each workflow to produce `imputation_report.json` (per‑column missing rates, sample matches).
- Runs `scripts/check_imputation_thresholds.py` and **fails the job** if thresholds are exceeded.
- If a failure occurs the workflow will:
  - Create a GitHub **Issue** describing the breach (uses the repository `GITHUB_TOKEN`).
  - Send a **Slack** message when `SLACK_WEBHOOK_URL` is configured.
  - POST the report to `IMPUTATION_TELEMETRY_URL` when configured.

Secrets / environment variables

- `GITHUB_TOKEN` — used by the workflow to create an issue (provided automatically by GitHub Actions).
- `SLACK_WEBHOOK_URL` — optional; when set, CI posts a short Slack alert on threshold breaches.
- `IMPUTATION_TELEMETRY_URL` — optional; CI and the running app will POST imputation events/reports to this endpoint.

Default thresholds (changeable)

- Row-level: `rows_with_any_imputation_prop` &lt; **0.10** (10%).
- Column-level: `per_column_missing_rate` &lt; **0.05** (5%).

You can override by editing `.github/workflows/ci.yml` or setting appropriate repository secrets.

How to run locally

- Generate the imputation report:

```bash
python monitor_imputation.py --artifact models/score_models.joblib --matches data/spi_matches.csv --stadiums data/stadium_coordinates.csv --recent-days 365 --out imputation_report.json
```

- Run the threshold checker:

```bash
python scripts/check_imputation_thresholds.py --report imputation_report.json --max-row-prop 0.10 --max-col-rate 0.05
```

Notes

- The Streamlit UI also logs imputation events locally to `logs/imputation_events.log` and will POST best‑effort telemetry when `IMPUTATION_TELEMETRY_URL` is set.
- Adjust thresholds to match your operational tolerance before enabling Slack/telemetry alerts.

