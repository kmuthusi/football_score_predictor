# Football Prediction App

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
