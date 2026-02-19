# Football Prediction App — Comprehensive Project Documentation

This document is the full technical and operational reference for the **Football Prediction App** project.

It describes:
- project purpose and architecture
- data contracts and feature engineering
- training pipeline, model artifact schema, and evaluation
- inference APIs and Streamlit application behavior
- reproducible workflows, testing, and troubleshooting

---

## 1) Project Overview

### 1.1 Objective

The project predicts football match outcomes at **exact scoreline** granularity by:

1. Estimating expected goals for both teams (`λ_home`, `λ_away`) using two Poisson regressors.
2. Converting those expected goals into a full scoreline probability matrix.
3. Optionally applying:
   - **Dixon-Coles low-score correction** (for 0-0, 1-0, 0-1, 1-1 interactions)
   - **Post-hoc temperature calibration** for scoreline probability reliability.

### 1.2 Core Use Cases

- Pre-match scoreline probability estimation
- W/D/L probability summary
- Model diagnostics and calibration tracking over time
- Experimental ablations on feature families and recency weighting

### 1.3 Repository Layout

```text
app.py                      # Streamlit UI for prediction and diagnostics
train.py                    # End-to-end training + evaluation + artifact save
predict.py                  # Inference utilities and probability transformations
features.py                 # Data preprocessing + leakage-safe feature engineering
metrics.py                  # Scoring metrics and Dixon-Coles fitting helpers
calibration_trends.py       # Trend helper for run-to-run calibration logs
config.py                   # Paths and default constants
requirements.txt            # Python dependencies
README.md                   # Quickstart-focused documentation
data/
  spi_matches.csv
  stadium_coordinates.csv
models/
  score_models.joblib
  low_score_calibration_log.csv
tests/
  test_features_contract.py
  test_predict_probs.py
```

---

## 2) Environment and Dependencies

### 2.1 Requirements

- Python 3.10+ recommended
- Dependencies (from `requirements.txt`):
  - `pandas>=2.0`
  - `numpy>=1.23`
  - `scikit-learn>=1.2`
  - `joblib>=1.2`
  - `streamlit>=1.30`

### 2.2 Installation

```bash
pip install -r requirements.txt
```

### 2.3 Default Paths

Defined in `config.py`:

- Matches CSV: `data/spi_matches.csv`
- Stadium coordinates CSV: `data/stadium_coordinates.csv`
- Models directory: `models`
- Artifact filename: `score_models.joblib`

---

## 3) Data Contracts

### 3.1 Matches Input (`spi_matches.csv`)

Expected fields used by the pipeline:

- `div` (league identifier)
- `date`
- `home_team`, `away_team`
- `home_goals`, `away_goals`
- `home_odds`, `draw_odds`, `away_odds`

Additional fields (e.g., `play_id`, `season`, `country`) can exist but are not mandatory for modeling.

### 3.2 Stadium Coordinates Input (`stadium_coordinates.csv`)

Expected fields:

- `team_name`
- `league`
- `latitude`
- `longitude`
- `status` (if absent, defaults to `success` in preprocessing)

Used to compute travel distance (`away_travel_km`) where coordinates are available.

---

## 4) Feature Engineering Pipeline

Implemented in `features.py` with leakage-safe temporal logic.

### 4.1 Design Principle

For a match at date **T**, historical form features are computed only from matches strictly **before T**.

### 4.2 Main Feature Groups

1. **Implied probability features from odds**
   - `p_home`, `p_draw`, `p_away`
   - `overround`

2. **Rolling form features** (windowed)
   - goals for/against and points (overall + location-specific)

3. **EWM form features**
   - exponentially weighted form summaries
   - controlled by `--use-ewm-features` / `--no-use-ewm-features`

4. **Opponent-adjusted features**
   - relative form by subtracting opponent prior strength proxies
   - controlled by `--use-adjusted-features` / `--no-use-adjusted-features`

5. **Travel distance (optional availability)**
   - `away_travel_km` based on stadium coordinates

### 4.3 Model Column Contracts

- Categorical columns are provided by `model_categorical_columns()`.
- Numeric columns are provided by `model_numeric_columns(config)`.
- `tests/test_features_contract.py` enforces that single-match feature generation returns all required model columns.

---

## 5) Modeling Approach

### 5.1 Base Model

Two independent `PoissonRegressor` models are trained:

- Home goals model (`model_home`)
- Away goals model (`model_away`)

Pipeline structure:

- categorical: impute most frequent + one-hot encoding
- numeric: median imputation + standard scaling
- estimator: Poisson regression with L2-style regularization (`alpha`)

### 5.2 Time-Based Split

`train.py` uses a strict chronological split:

- `max_date` from available data
- `cutoff = max_date - test_days`
- training rows: date `< cutoff`
- testing rows: date `>= cutoff`

Default test window is `365` days.

### 5.3 Time-Decay Weighting

Optional exponential sample weighting in train split:

- weight formula: `w = 0.5 ** (delta_days / half_life_days)`
- set `--decay-half-life-days 0` to disable
- optional tuning via `--tune-decay` and candidate half-lives

---

## 6) Calibration and Scientific Controls

### 6.1 Dixon-Coles Low-Score Correction

Enabled with `--fit-dc`.

- Fits global `rho`
- Optionally fits per-league `rho` if enough league rows (`--dc-min-league-matches`)
- Search bounds and grid granularity are configurable:
  - `--dc-rho-min`, `--dc-rho-max`
  - `--dc-coarse-steps`, `--dc-fine-steps`

### 6.2 OOT Significance Selection for Rho

When decay tuning uses `--tune-metric dc_nll` and `--dc-optimize-oot`, the selected `rho` can be chosen from out-of-time validation and compared against `rho=0` using a paired significance-style criterion (`--dc-significance-z`).

### 6.3 Scoreline Temperature Calibration

Enabled by default (`--fit-score-calibration`, disable with `--no-fit-score-calibration`).

- Fits temperature on internal validation slice (`--calibration-val-days`)
- Optional per-league temperatures:
  - `--score-calibration-by-league`
  - `--calibration-min-league-rows`

### 6.4 Strict Scientific Mode Flag

Artifact field `scientific_mode.strict_scientific_mode` becomes true under a strict combination of settings (DC + decay tuning + OOT optimization + calibration + no concentration cap heuristic).

### 6.5 Model Selection Policy (Current)

Current project policy for low-score handling:

- Keep **Dixon-Coles** as the primary low-score correction method.
- Only introduce a **low-score-inflated mixture** extension if held-out diagnostics indicate persistent underfit after DC + calibration.

Underfit trigger criteria should be based on out-of-time evidence, including:

- no meaningful held-out improvement in `NLL`/`ECE` from additional calibration steps, and
- systematic low-score residual gaps (notably `0-0`, `1-0`, `0-1`, `1-1`).

---

## 7) Training CLI Reference

Run training:

```bash
python train.py --matches data/spi_matches.csv --stadiums data/stadium_coordinates.csv --out models
```

### 7.1 Core Arguments

- `--matches` (str): matches CSV path
- `--stadiums` (str): stadium CSV path
- `--out` (str): output directory
- `--test-days` (int, default 365)
- `--windows` (int list, default `5 10`)
- `--max-goals` (int, default 6)
- `--alpha` (float, default `1e-4`)
- `--max-iter` (int, default 500)

### 7.2 Recency / Decay Arguments

- `--decay-half-life-days` (float, default 0)
- `--tune-decay`
- `--decay-candidates` (float list)
- `--val-days` (int)
- `--tune-metric` (`ind_nll` or `dc_nll`)

### 7.3 Feature Ablation Arguments

- `--use-ewm-features` / `--no-use-ewm-features`
- `--use-adjusted-features` / `--no-use-adjusted-features`

### 7.4 Dixon-Coles Arguments

- `--fit-dc`
- `--dc-rho-min`, `--dc-rho-max`
- `--dc-coarse-steps`, `--dc-fine-steps`
- `--dc-min-league-matches`
- `--dc-optimize-oot`
- `--dc-significance-z`
- `--dc-max-top-share`

### 7.5 Scoreline Calibration Arguments

- `--fit-score-calibration` / `--no-fit-score-calibration`
- `--calibration-val-days`
- `--score-calibration-by-league` / `--no-score-calibration-by-league`
- `--calibration-min-league-rows`
- `--fit-low-score-mixture` / `--no-fit-low-score-mixture` (minimal low-score mixture fitted on calibration split)

### 7.6 Diagnostics Arguments

- `--backtest-folds` (rolling-origin stability diagnostics)

---

## 8) Evaluation Outputs

`train.py` prints and records:

- expected-goals regression metrics
  - MAE, RMSE, Poisson deviance (home and away)
- exact score top-1 accuracy
- W/D/L accuracy
- average NLL (independent Poisson)
- average NLL (Dixon-Coles) when enabled
- benchmark panel vs simple baselines
- Diebold-Mariano-style comparison (model vs empirical baseline)
- lambda predictive interval diagnostics
- optional rolling-origin backtest metrics
- low-score calibration summary for 0-0, 1-0, 0-1, 1-1

Run summaries are appended to:

- `models/low_score_calibration_log.csv`

---

## 9) Artifact Schema (`models/score_models.joblib`)

Top-level keys include:

- `model_home`, `model_away`
- `config`
- `cat_cols`, `num_cols`
- split metadata (`train_rows`, `test_rows`, `cutoff_date`, `max_date`)
- backward-compatible `dixon_coles_rho`
- `dixon_coles` block (rho source, bounds, per-league rhos, guardrail info)
- `time_decay` block
- `decay_tuning` block
- `scientific_mode` block
- `scoreline_calibration` block
- `diagnostics` block

This artifact is consumed by both `predict.py` and `app.py`.

---

## 10) Inference Utilities (`predict.py`)

### 10.1 Main Functions

- `load_artifact(path)`
- `predict_expected_goals(artifact, X)`
- `scoreline_probability_matrix(lam_home, lam_away, max_goals, include_tail_bucket, rho)`
- `calibrate_scoreline_matrix(mat, calibration_cfg, div)`
- `win_draw_loss_probs(lam_home, lam_away, max_k, rho)`
- `top_scorelines(mat, top_n)`

### 10.2 Probability Logic

- Base matrix uses independent Poisson assumptions.
- Optional Dixon-Coles modifies low-score cells then renormalizes.
- Optional temperature scaling is applied to flattened matrix probabilities then reshaped and renormalized.

### 10.3 Tail Bucket

When `include_tail_bucket=True`, the matrix contains an extra `max_goals+` row/column to preserve probability mass beyond truncation.

---

## 11) Streamlit Application (`app.py`)

Launch:

```bash
streamlit run app.py
```

### 11.1 App Responsibilities

- Loads trained artifact and source data
- Collects match inputs (league/teams/date/odds)
- Builds single-match feature row
- Predicts expected goals and scoreline distributions
- Displays:
  - top scorelines
  - scoreline probability grid
  - optional Poisson vs DC comparison
  - artifact diagnostics and scientific metadata

### 11.2 Key UI Controls

- Artifact/data path setup (advanced panel)
- Optional Dixon-Coles checkbox + rho slider (if available in artifact)
- Grid color scale selector
- Concentration warning threshold

---

## 12) Reproducible Workflow

### 12.1 End-to-End Baseline

```bash
python train.py --matches data/spi_matches.csv --stadiums data/stadium_coordinates.csv --out models --max-iter 1000
streamlit run app.py
```

### 12.2 Full Recommended Enhanced Run

```bash
python train.py --matches data/spi_matches.csv --stadiums data/stadium_coordinates.csv --out models --fit-dc --tune-decay --tune-metric dc_nll --dc-optimize-oot --decay-candidates 0 365 730 --val-days 180 --max-iter 1000 --use-ewm-features --use-adjusted-features --max-goals 9
```

### 12.3 Calibration Trend Inspection

```bash
python -m calibration_trends --last-n 10
```

---

## 13) Testing

Current tests verify:

- feature contract for single-match prediction row (`tests/test_features_contract.py`)
- probability matrix normalization with and without Dixon-Coles (`tests/test_predict_probs.py`)

Run tests:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## 14) Operational Notes and Troubleshooting

### 14.1 Common Issues

For a quick artifact/version mismatch fix path, see **README → “Artifact compatibility (troubleshooting)”**.

1. **Artifact not found in app**
   - Retrain with `train.py` and verify `models/score_models.joblib` exists.

2. **Poisson models collapse to intercept-only**
   - Increase `--max-iter` (e.g., 1000).
   - Verify input data quality and finite feature values.

3. **Calibration skipped**
   - Internal calibration split may be too small; increase data size or adjust split windows.

4. **`dc_nll` tuning error**
   - `--tune-metric dc_nll` requires `--fit-dc`.

5. **Editor/runtime environment mismatch**
   - Ensure VS Code interpreter points to environment where dependencies were installed.

### 14.2 Data Quality Recommendations

- Validate odds are positive and present when possible.
- Ensure dates parse cleanly and are timezone-consistent.
- Keep team naming consistent between matches and stadium coordinates.

---

## 15) Performance and Scaling Considerations

- Training cost scales with number of historical rows and engineered feature volume.
- Feature frame generation is often the dominant preprocessing cost.
- Backtesting (`--backtest-folds`) and DC tuning/calibration add runtime.
- If iterating rapidly, reduce candidate grids/folds first, then restore for final runs.

---

## 16) Extensibility Points

Potential future extensions can be added without redesigning the current architecture:

- alternative calibration methods beyond temperature scaling
- richer uncertainty quantification per match
- hyperparameter search wrappers around current CLI
- model registry/versioning for artifacts
- scheduled retraining and monitoring workflows

---

## 17) Quick Reference Commands

```bash
# Install deps
pip install -r requirements.txt

# Train baseline
python train.py --matches data/spi_matches.csv --stadiums data/stadium_coordinates.csv --out models

# Train enhanced configuration
python train.py --matches data/spi_matches.csv --stadiums data/stadium_coordinates.csv --out models --fit-dc --tune-decay --tune-metric dc_nll --dc-optimize-oot --decay-candidates 0 365 730 --val-days 180 --max-iter 1000 --use-ewm-features --use-adjusted-features

# Launch app
streamlit run app.py

# View calibration trends
python -m calibration_trends --last-n 10

# Run tests
python -m unittest discover -s tests -p "test_*.py"
```

---

## 18) Document Maintenance

When behavior changes in any of these files, update this document in the same pull request:

- `train.py` (CLI, diagnostics, artifact schema)
- `predict.py` (inference and probability logic)
- `features.py` (feature contracts)
- `app.py` (UI behavior and controls)
- `requirements.txt` (dependency constraints)

This keeps operational documentation aligned with implementation.
