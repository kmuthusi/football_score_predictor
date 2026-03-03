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
evaluate_reproducible.py    # Reproducible held-out evaluation CLI
monitor_imputation.py       # Imputation monitoring + CI report generation
rl_env.py / rl_train.py / rl_eval.py  # Optional RL betting policy stack
probability_utils.py        # Shared probability helper functions (W/D/L, implied probs)
config.py                   # Paths and default constants
requirements.txt            # Python dependencies
README.md                   # Quickstart-focused documentation
data/
  spi_matches.csv
  stadium_coordinates_completed_full.csv
models/
  score_models.joblib
  low_score_calibration_log.csv
tests/
   test_predict_preprocessing.py
   test_monitor_imputation.py
   test_check_imputation_thresholds.py
   test_low_score_features.py
   test_concentration_shrinker.py
   test_rl_env.py
   test_rl_train_smoke.py
   test_rl_eval_smoke.py
   test_rl_policy_safety.py
   test_scientific_checks.py
```

---

## 2) Environment and Dependencies

### 2.1 Requirements

- Python 3.10+ recommended
- Dependencies (from `requirements.txt`):
   - `numpy>=1.26,<3.0`
   - `pandas>=2.1,<3.0`
   - `scipy>=1.11,<2.0`
   - `scikit-learn==1.6.1`
   - `joblib>=1.3,<2.0`
   - `streamlit>=1.35,<2.0`

### 2.2 Installation

```bash
pip install -r requirements.txt
```

### 2.3 Default Paths

Defined in `config.py`:

- Matches CSV: `data/spi_matches.csv`
- Stadium coordinates CSV: `data/stadium_coordinates_completed_full.csv`
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

### 3.2 Stadium Coordinates Input (`stadium_coordinates_completed_full.csv`)

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

6. **Rest and Travel Interaction (NEW)**
   - `home_rest_travel_interaction`: home team rest days × away team travel distance
   - `away_rest_travel_interaction`: away team rest days × away team travel distance
   - Captures combined effect of team fatigue and travel distance
   - Improves model's ability to explain situational home advantage

### 4.3 Model Column Contracts

- Categorical columns are provided by `model_categorical_columns()`.
- Numeric columns are provided by `model_numeric_columns(config)`.
- `tests/test_predict_preprocessing.py` enforces single-match feature generation column parity with artifact expectations.

---

## 5) Modeling Approach

### 5.1 Base Model

Two independent regression models are trained for goal counts:

- Home goals model (`model_home`)
- Away goals model (`model_away`)

**Default: Poisson Regression**
- `PoissonRegressor` from scikit-learn
- Assumes mean equals variance (equidispersion)
- Good for well-calibrated count data

**Alternative: Negative Binomial (Tweedie) - EXPERIMENTAL**
- `TweedieRegressor(power=1.5)` handles overdispersion
- Poisson-Gamma compound distribution
- Better for zero-inflated goal data with excess variance
- Enable with `--use-negative-binomial` flag

Pipeline structure:

- categorical: impute most frequent + one-hot encoding  
- numeric: median imputation + standard scaling
- estimator: Poisson or Tweedie regression with L2-style regularization (`alpha`)

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

### 6.3 Isotonic Regression Calibration (NEW)

Enabled with `--use-isotonic-calibration`.

- Fits `IsotonicRegression` from scikit-learn on validation set
- Non-parametric, threshold-sensitive probability adjustment
- Applied **after** temperature scaling for fine-tuning
- Improves NLL (negative log-likelihood) on calibration data
- Mechanism:
  1. Build probability matrix for each match in validation set
  2. Flatten to binary targets (one-hot per scoreline outcome)
  3. Fit isotonic regressor: `isotonic_reg.predict(flat_probs)`
  4. Store regressor in artifact for inference-time application
- At inference: applies isotonic transformation to calibrated probabilities
- Recommendation: Start with isotonic-only, then test with other enhancements

### 6.4 Scoreline Temperature Calibration

Enabled by default (`--fit-score-calibration`, disable with `--no-fit-score-calibration`).

- Fits temperature on internal validation slice (`--calibration-val-days`)
- Optional per-league temperatures:
  - `--score-calibration-by-league`
  - `--calibration-min-league-rows`

### 6.5 Strict Scientific Mode Flag

Artifact field `scientific_mode.strict_scientific_mode` becomes true under a strict combination of settings (DC + decay tuning + OOT optimization + calibration + no concentration cap heuristic).

### 6.6 Model Selection Policy (Current)

Current project policy for low-score handling:

- Keep **Dixon-Coles** as the primary low-score correction method.
- Only introduce a **low-score-inflated mixture** extension if held-out diagnostics indicate persistent underfit after DC + calibration.

Underfit trigger criteria should be based on out-of-time evidence, including:

- no meaningful held-out improvement in `NLL`/`ECE` from additional calibration steps, and
- systematic low-score residual gaps (notably `0-0`, `1-0`, `0-1`, `1-1`).
The low-score pipeline now also includes:

- `--fit-low-score-mixture` / `--no-fit-low-score-mixture` flags to optionally fit a small mixture component targeting total goals &lt;=1 during calibration training.
- `--low-score-alpha` (float) to blend mixture probability with base model (mirrors `scoreline_calibration.low_score_alpha` in the artifact).

A companion analysis script [`scripts/low_score_analysis.py`] computes empirical vs model low-score rates and writes a JSON report.  CI consumes that report via `scripts/check_low_score_bias.py` (threshold default 0.02) and will fail when the discrepancy is large.

---
---

## 7) Training CLI Reference

Run training:

```bash
python train.py --matches data/spi_matches.csv --stadiums data/stadium_coordinates_completed_full.csv --out models
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

### 7.2 Model Selection Arguments (NEW)

- `--use-negative-binomial` / `--no-use-negative-binomial` (default False)
  - Use `TweedieRegressor(power=1.5)` instead of `PoissonRegressor`
  - Handles overdispersion in goal count data
  - Experimental; compare against baseline before production

- `--use-isotonic-calibration` / `--no-use-isotonic-calibration` (default False)
  - Fit isotonic regression for post-temperature probability calibration
  - Applied after temperature scaling
  - Improves NLL on validation data
  - Recommended for accuracy improvement

### 7.3 Recency / Decay Arguments

- `--decay-half-life-days` (float, default 0)
- `--tune-decay`
- `--decay-candidates` (float list)
- `--val-days` (int)
- `--tune-metric` (`ind_nll` or `dc_nll`)

### 7.4 Feature Ablation Arguments

- `--use-ewm-features` / `--no-use-ewm-features`
- `--use-adjusted-features` / `--no-use-adjusted-features`

### 7.5 Dixon-Coles Arguments

- `--fit-dc`
- `--dc-rho-min`, `--dc-rho-max`
- `--dc-coarse-steps`, `--dc-fine-steps`
- `--dc-min-league-matches`
- `--dc-optimize-oot`
- `--dc-significance-z`
- `--dc-max-top-share`

### 7.6 Scoreline Calibration Arguments

- `--fit-score-calibration` / `--no-fit-score-calibration`
- `--calibration-val-days`
- `--score-calibration-by-league` / `--no-score-calibration-by-league`
- `--calibration-min-league-rows`
- `--fit-low-score-mixture` / `--no-fit-low-score-mixture` (minimal low-score mixture fitted on calibration split)

### 7.7 Diagnostics Arguments

- `--backtest-folds` (rolling-origin stability diagnostics)

### 7.8 Reinforcement Learning Policy \(Optional\)

An optional RL module trains a lightweight policy to place (or skip) bets based on model outputs and live odds.  It is **separate** from the core scoreline predictor; the policy takes the same features plus a single low-score probability and log-bankroll.

- Environment defined in `rl_env.py` and training/ evaluation logic in `rl_train.py`/`rl_eval.py`.
- Observations (14-dim) consist of:
  1. raw W/D/L probabilities (`pH`, `pD`, `pA`)
  2. implied probabilities from the market (`impH`, `impD`, `impA`)
  3. edges (`edgeH`, `edgeD`, `edgeA`)
  4. raw odds (`ho`, `do`, `ao`)
  5. log(bankroll) and `low1` probability (total goals &lt;=1)

Training configuration mirrors `EnvConfig` and `TrainConfig` with arguments:

- `--initial-bankroll`, `--stake-frac`, `--max-stake-frac` etc. for sizing
- `--ev-threshold`, `--min-bankroll`, `--min-stake` for safety gating
- `--bet-penalty` (per-bet cost) and `--low-score-penalty` (penalty proportional to predicted low1 probability)
- `--reward-mode` (`log_growth` or `profit`)

Safety features included in the pipeline:

1. **EV gating** prevents actions with negative expected value.
2. **Min stake / bankroll** early-stop criteria.
3. **Bet penalty** discourages over-trading.
4. **Low-score penalty** reduces reward on potentially unfair low-scoring fixtures.
5. **Artifact metadata validation:** `scripts/check_rl_policy_safety.py` ensures that a saved policy contains required configuration and observation dimensionality before it's used in production.

Training example:

```bash
python rl_train.py --artifact models/score_models.joblib --matches data/spi_matches.csv \
    --stadiums data/stadium_coordinates_completed_full.csv --epochs 30 \
    --low-score-penalty 0.05 --bet-penalty 0.01
```

Evaluation:

```bash
python rl_eval.py --policy models/rl_policy.joblib --artifact models/score_models.joblib \
    --matches data/spi_matches.csv --stadiums data/stadium_coordinates_completed_full.csv
```
#### ⚠️ Diagnostic-Only: RL Policy is Not Profitable

**Critical limitation:** The RL betting policy is provided for **educational and diagnostic purposes only**. **Do not use it for real money betting.**

Testing shows that all policy configurations (across multiple bet_penalty and ev_threshold combinations) achieve **negative ROI** on backtests:

| Configuration | EV Threshold | Bet Penalty | ROI (Staked) | Bets |
|---|---|---|---|---|
| Default (main) | 0.01 | 0.01 | -5.35% | 5,654 |
| Strict (ev=0.05) | 0.05 | 0.001 | -9.16% | 3,652 |
| Loose (ev=0.00) | 0.00 | 0.001 | -6.61% | 6,348 |

**Why this happens:**
1. The scoreline model is well-calibrated (49.92% W/D/L accuracy, good NLL) but lacks persistent market edge
2. Professional bookmakers set odds using advanced ML and sharp consensus
3. Free public odds data reflects consensus probability, not exploitable edge
4. Beating efficient markets is extremely difficult and rare

**Appropriate uses:**
- **Learning:** understand policy-gradient REINFORCE training
- **Diagnostics:** identify high-EV matches according to the model
- **Simulation:** test bankroll management strategies without real stakes
- **Dashboard:** Streamlit app displays predicted W/D/L probabilities (valuable for analysis)

**Do not use for:**
- Real-money betting
- Production automated trading
- Any deployment expecting positive returns

The Streamlit prediction dashboard is still valuable for analytical purposes (W/D/L predictions, calibration diagnostics, bet-finding via manual review).
CI also runs a lightweight “smoke” train+eval job to ensure RL code remains functional.

---

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
- `config` (includes `use_isotonic_calibration` and `use_negative_binomial` flags)
- `cat_cols`, `num_cols`
- split metadata (`train_rows`, `test_rows`, `cutoff_date`, `max_date`)
- backward-compatible `dixon_coles_rho`
- `dixon_coles` block (rho source, bounds, per-league rhos, guardrail info)
- `time_decay` block
- `decay_tuning` block
- `scientific_mode` block
- `scoreline_calibration` block:
  - `temperature` and `temperature_by_league` (if enabled)
  - `isotonic_regressor` (if `--use-isotonic-calibration` enabled, sklearn object)
  - `low_score_alpha` and low-score mixture weights
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

- Base matrix uses independent Poisson (or Tweedie if enabled) assumptions.
- Optional Dixon-Coles modifies low-score cells then renormalizes.
- Optional temperature scaling is applied to flattened matrix probabilities then reshaped and renormalized.
- Optional isotonic regression (NEW): applied post-temperature-scaling to fine-tune probabilities:
  1. Flatten calibrated probability matrix
  2. Apply isotonic regressor (if available)
  3. Reshape and renormalize
  4. Improves reliability without changing rank order significantly

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
python train.py --matches data/spi_matches.csv --stadiums data/stadium_coordinates_completed_full.csv --out models --max-iter 1000
streamlit run app.py
```

### 12.2 Full Recommended Enhanced Run

```bash
python train.py --matches data/spi_matches.csv --stadiums data/stadium_coordinates_completed_full.csv --out models --fit-dc --tune-decay --tune-metric dc_nll --dc-optimize-oot --decay-candidates 0 365 730 --val-days 180 --max-iter 1000 --use-ewm-features --use-adjusted-features --max-goals 9
```

### 12.3 Model Enhancements (v2.0)

**Quick comparison runs to test new improvements:**

```bash
# Recommended: Isotonic calibration only (safest, fastest)
python train.py --matches data/spi_matches.csv --stadiums data/stadium_coordinates_completed_full.csv --out models \
  --fit-dc --use-isotonic-calibration --max-iter 500

# All three enhancements (maximum potential)
python train.py --matches data/spi_matches.csv --stadiums data/stadium_coordinates_completed_full.csv --out models \
  --fit-dc --use-isotonic-calibration --use-negative-binomial --max-iter 500

# Negative binomial only
python train.py --matches data/spi_matches.csv --stadiums data/stadium_coordinates_completed_full.csv --out models \
  --fit-dc --use-negative-binomial --max-iter 500

# Baseline (for comparison)
python train.py --matches data/spi_matches.csv --stadiums data/stadium_coordinates_completed_full.csv --out models \
  --fit-dc --max-iter 500
```

### 12.4 Calibration Trend Inspection

```bash
python -m calibration_trends --last-n 10
```

---

## 13) Testing

Current tests verify:

- prediction preprocessing column parity (`tests/test_predict_preprocessing.py`)
- imputation monitoring and threshold guard behavior (`tests/test_monitor_imputation.py`, `tests/test_check_imputation_thresholds.py`)
- low-score feature engineering and concentration shrink guard (`tests/test_low_score_features.py`, `tests/test_concentration_shrinker.py`)
- RL environment/train/eval/safety smoke checks (`tests/test_rl_env.py`, `tests/test_rl_train_smoke.py`, `tests/test_rl_eval_smoke.py`, `tests/test_rl_policy_safety.py`)
- statistical/scientific diagnostics sanity checks (`tests/test_scientific_checks.py`)

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
python train.py --matches data/spi_matches.csv --stadiums data/stadium_coordinates_completed_full.csv --out models

# Train with new enhancements (recommended)
python train.py --matches data/spi_matches.csv --stadiums data/stadium_coordinates_completed_full.csv --out models \
  --fit-dc --use-isotonic-calibration --max-iter 500

# Train with all three model improvements
python train.py --matches data/spi_matches.csv --stadiums data/stadium_coordinates_completed_full.csv --out models \
  --fit-dc --use-isotonic-calibration --use-negative-binomial --max-iter 500

# Train full enhanced configuration
python train.py --matches data/spi_matches.csv --stadiums data/stadium_coordinates_completed_full.csv --out models \
  --fit-dc --tune-decay --tune-metric dc_nll --dc-optimize-oot --decay-candidates 0 365 730 \
  --val-days 180 --max-iter 1000 --use-ewm-features --use-adjusted-features \
  --use-isotonic-calibration --use-negative-binomial

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

- `train.py` (CLI, model selection, diagnostics, artifact schema, enhancements)
- `predict.py` (inference and probability logic, calibration pipeline)
- `features.py` (feature contracts, interaction features)
- `metrics.py` (scoring metrics, calibration functions)
- `app.py` (UI behavior and controls)
- `requirements.txt` (dependency constraints)

**v2.0 enhancements (added March 2026):**
- Isotonic regression calibration (NEW)
- Rest/travel interaction features (NEW)
- Negative binomial regressor option (NEW)

This keeps operational documentation aligned with implementation.
