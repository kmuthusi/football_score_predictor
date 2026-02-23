import sys
from pathlib import Path
# ensure project root is on sys.path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import sklearn.compose._column_transformer as ct
class _RemainderColsList(list): pass
ct._RemainderColsList = _RemainderColsList

import pandas as pd, numpy as np
from predict import load_artifact, predict_expected_goals, scoreline_probability_matrix, calibrate_scoreline_matrix
from features import FeatureConfig, build_training_frame, load_matches_csv, load_stadiums_csv

def main():
    print('loading matches...')
    matches = load_matches_csv('data/spi_matches.csv')
    print('loading artifact...')
    artifact = load_artifact('models/score_models.joblib')
    stadiums = load_stadiums_csv('data/stadium_coordinates.csv')
    print('building frame...')
    cfg = FeatureConfig(**artifact.get('config', {}))
    frame = build_training_frame(matches, stadiums, cfg)
    frame = frame.dropna(subset=['home_goals','away_goals']).copy()
    print('frame rows', len(frame))

    cat_cols=list(artifact.get('cat_cols',[]))
    num_cols=list(artifact.get('num_cols',[]))
    X = frame[cat_cols+num_cols]
    print('predicting expected goals...')
    lam_h, lam_a = predict_expected_goals(artifact, X)
    lam_h = np.asarray(lam_h,dtype=float)
    lam_a = np.asarray(lam_a,dtype=float)
    cal_cfg = artifact.get('scoreline_calibration')
    print('computing low score probabilities...')
    pred_low1=[]
    pred_low2=[]
    for i,r in frame.iterrows():
        lamhi,lamai = lam_h[i], lam_a[i]
        mat=scoreline_probability_matrix(lamhi,lamai,max_goals=int(artifact.get('config',{}).get('max_goals',6)),include_tail_bucket=True,rho=None)
        mat=calibrate_scoreline_matrix(mat,cal_cfg,div=str(r['div']) if pd.notna(r['div']) else None)
        arr=mat.to_numpy(float)
        if arr.shape[0]>1:
            low1=float(arr[0,0]+arr[0,1]+arr[1,0])
            low2=float(low1+arr[0,2]+arr[2,0]+arr[1,1])
        else:
            low1=float(arr[0,0]); low2=low1
        pred_low1.append(low1); pred_low2.append(low2)
    pred_low1=np.array(pred_low1)
    pred_low2=np.array(pred_low2)
    tot_goals=frame['home_goals']+frame['away_goals']
    obs_low1=(tot_goals<=1).astype(float)
    obs_low2=(tot_goals<=2).astype(float)
    obs1 = float(obs_low1.mean())
    pred1 = float(pred_low1.mean())
    bias1 = float(np.mean(pred_low1 - obs_low1))
    obs2 = float(obs_low2.mean())
    pred2 = float(pred_low2.mean())
    bias2 = float(np.mean(pred_low2 - obs_low2))
    print(f"obs low1 {obs1}")
    print(f"pred low1 mean {pred1}")
    print(f"bias low1 {bias1}")
    print(f"obs low2 {obs2}")
    print(f"pred low2 {pred2}")
    print(f"bias low2 {bias2}")

    # write JSON summary for downstream checks
    out = {
        "obs_low1": obs1,
        "pred_low1": pred1,
        "bias_low1": bias1,
        "obs_low2": obs2,
        "pred_low2": pred2,
        "bias_low2": bias2,
    }
    import json, os
    os.makedirs("reports", exist_ok=True)
    with open("reports/low_score_bias.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Wrote reports/low_score_bias.json")


if __name__ == "__main__":
    main()
