"""Check saved RL policy payload for required safety metadata.

Exits non-zero when the policy is missing required fields (bet_penalty,
ev_threshold, and obs_norm when use_obs_norm=True). This is intended for
CI use so the pipeline can reject unsafe policy artifacts.
"""
import argparse
import sys
import joblib
from pathlib import Path

# allow import from repository root when invoked from scripts/ directory
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from rl_policy_utils import policy_is_safe


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", type=str, default="models/rl_policy_test.joblib")
    args = p.parse_args(argv)

    try:
        payload = joblib.load(args.policy)
    except Exception as e:
        print(f"[error] failed to load policy: {e}")
        return 2

    ok, reason = policy_is_safe(payload)
    if not ok:
        print(f"[unsafe] RL policy failed safety checks: {reason}")
        return 1

    print("[ok] RL policy safety checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
