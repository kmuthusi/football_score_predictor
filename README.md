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

## Notes

- If the app cannot find the artifact, retrain and confirm `models/score_models.joblib` exists.
- If convergence is weak, increase `--max-iter`.
- Odds inputs are optional but typically improve signal quality.
