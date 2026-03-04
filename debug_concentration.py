"""Debug script to understand 1-1 concentration issue"""
import joblib
import numpy as np
from predict import scoreline_probability_matrix, calibrate_scoreline_matrix
from features import load_matches_csv, build_training_frame, load_stadiums_csv
import pandas as pd

np.set_printoptions(precision=3, suppress=True)

artifact = joblib.load('models/score_models.joblib')
config = artifact['config'] if isinstance(artifact.get('config'), dict) else artifact['config'].__dict__

print("=== TRAINING CONFIG ===")
print(f"Dixon-Coles rho: {artifact.get('dixon_coles_rho', 'N/A')}")
max_goals = config.get('max_goals', 6) if isinstance(config, dict) else getattr(config, 'max_goals', 6)
use_isotonic = config.get('use_isotonic_calibration', False) if isinstance(config, dict) else getattr(config, 'use_isotonic_calibration', False)
use_nb = config.get('use_negative_binomial', False) if isinstance(config, dict) else getattr(config, 'use_negative_binomial', False)
print(f"Use Isotonic: {use_isotonic}")
print(f"Use Negative Binomial: {use_nb}")
print(f"Max goals: {max_goals}")
print(f"Test rows: {artifact.get('test_rows')}")

# Check calibration
sc_cal = artifact.get('scoreline_calibration', {})
if 'temperatures_by_div' in sc_cal:
    temps = sc_cal['temperatures_by_div']
    print(f"\n=== TEMPERATURE CALIBRATION ===")
    print(f"Number of leagues: {len(temps)}")
    temps_list = list(temps.values())
    print(f"Temperature range: {np.min(temps_list):.3f} to {np.max(temps_list):.3f}")
    print(f"Mean temperature: {np.mean(temps_list):.3f}")

if 'isotonic' in sc_cal:
    iso = sc_cal['isotonic']
    print(f"\n=== ISOTONIC CALIBRATION ===")
    print(f"NLL improvement: {iso.get('nll_improvement', 'N/A'):.6f}")
    print(f"Training rows: {iso.get('n_rows', 'N/A')}")

# Load test data and check expected goals distribution
print("\n=== EXPECTED GOALS ANALYSIS ===")
matches = load_matches_csv('data/spi_matches.csv')
stadiums = load_stadiums_csv('data/stadium_coordinates_completed_full.csv')

cutoff = artifact['cutoff_date']
print(f"Cutoff date: {cutoff}")

# Prepare config for feature building
from features import FeatureConfig
fc = FeatureConfig()

# Build frame for the ENTIRE dataset then filter to test
df_train = build_training_frame(matches, stadiums, config=fc)
df_test = df_train[df_train['date'] >= cutoff].reset_index(drop=True)

print(f"Test set size: {len(df_test)} matches")
test_indices = list(range(min(100, len(df_test))))  # Sample first 100
test_sample = df_test.iloc[test_indices].copy()

model_home = artifact['model_home']
model_away = artifact['model_away']

# Get feature columns
cat_cols = artifact['cat_cols']
num_cols = artifact['num_cols']
feature_cols = list(cat_cols) + list(num_cols)

# Drop non-feature columns for prediction
X_test = test_sample[feature_cols]
lam_home = model_home.predict(X_test)
lam_away = model_away.predict(X_test)

print(f"\nExpected goals (sample of {len(lam_home)} matches):")
print(f"  Home: mean={np.mean(lam_home):.3f}, std={np.std(lam_home):.3f}, min={np.min(lam_home):.3f}, max={np.max(lam_home):.3f}")
print(f"  Away: mean={np.mean(lam_away):.3f}, std={np.std(lam_away):.3f}, min={np.min(lam_away):.3f}, max={np.max(lam_away):.3f}")

# Check what scorelines are being predicted
rho = artifact.get('dixon_coles_rho')
mode_scores = []
for i in range(min(100, len(lam_home))):
    mat = scoreline_probability_matrix(
        float(lam_home[i]), float(lam_away[i]), 
        max_goals=int(max_goals), 
        include_tail_bucket=True, 
        rho=float(rho) if rho else None
    )
    div = test_sample.iloc[i]['div']
    mat = calibrate_scoreline_matrix(mat, sc_cal, div=str(div) if div else None)
    arr = mat.to_numpy()
    mode_idx = np.argmax(arr)
    r, c = divmod(mode_idx, arr.shape[1])
    mode_score = f"{mat.index[r]}-{mat.columns[c]}"
    mode_scores.append(mode_score)

from collections import Counter
score_counts = Counter(mode_scores)
print(f"\nTop predicted scorelines (sample of {len(mode_scores)} matches):")
for score, count in score_counts.most_common(10):
    pct = 100 * count / len(mode_scores)
    print(f"  {score}: {count} ({pct:.1f}%)")
