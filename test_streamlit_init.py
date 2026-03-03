import sys
print("Step 1: Loading artifact path...", flush=True)
from config import DEFAULT_ARTIFACT_ABS_PATH
artifact_path = DEFAULT_ARTIFACT_ABS_PATH
print(f"  Artifact path: {artifact_path}", flush=True)

from pathlib import Path
print(f"Step 2: Checking if artifact exists...", flush=True)
if not Path(artifact_path).exists():
    print(f"  ERROR: Artifact not found at {artifact_path}", flush=True)
    sys.exit(1)
else:
    print(f"  Found: {artifact_path}", flush=True)

print(f"Step 3: Loading artifact...", flush=True)
from predict import load_artifact
artifact = load_artifact(artifact_path)
print(f"  Loaded! Keys: {list(artifact.keys())[:5]}", flush=True)

print(f"Step 4: Loading matches and stadiums...", flush=True)
from features import load_matches_csv, load_stadiums_csv
from config import DEFAULT_MATCHES_ABS_PATH, DEFAULT_STADIUMS_ABS_PATH
matches = load_matches_csv(DEFAULT_MATCHES_ABS_PATH)
stadiums = load_stadiums_csv(DEFAULT_STADIUMS_ABS_PATH)
print(f"  Loaded! Matches shape: {matches.shape}, Stadiums shape: {stadiums.shape}", flush=True)

print("All initialization steps completed successfully!")
