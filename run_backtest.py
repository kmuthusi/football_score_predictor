import subprocess
import sys

cmd = [
    sys.executable, "rl_eval.py",
    "--policy", "models/rl_policy.joblib",
    "--initial-bankroll", "10000",
    "--stake-frac", "0.01",
    "--max-stake-frac", "0.05",
    "--ev-threshold", "0.01",
]

print(f"Running: {' '.join(cmd)}\n")
result = subprocess.run(cmd, capture_output=False)
sys.exit(result.returncode)
