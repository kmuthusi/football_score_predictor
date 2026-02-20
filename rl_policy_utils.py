"""Utility helpers for validating RL policy payloads.

These helpers are lightweight and import-safe so the Streamlit app can
call them at runtime and the tests can import them without starting the UI.
"""
from typing import Tuple, Dict, Any


def policy_is_safe(policy: Dict[str, Any]) -> Tuple[bool, str]:
    """Basic safety / compliance checks for a saved RL policy payload.

    Returns (is_safe, reason). ``is_safe`` is True when the policy contains
    the minimal safety metadata we expect for production use. ``reason`` is an
    explanatory string when the policy is unsafe (empty when safe).

    Rules enforced (conservative):
    - policy must include a ``train_cfg`` mapping.
    - ``train_cfg`` must define non-null ``bet_penalty`` and ``ev_threshold``.
    - if ``use_obs_norm`` was True at train time, the payload must include
      ``obs_norm`` with both ``mean`` and ``std`` arrays.

    These checks are intentionally strict — they **flag** older policies so
    the app can avoid recommending risky bets from policies trained without
    the newer safety controls.
    """
    if not isinstance(policy, dict) or not policy:
        return False, "policy payload missing or invalid"

    train_cfg = (policy.get("train_cfg") or {})
    if not isinstance(train_cfg, dict) or not train_cfg:
        return False, "missing or invalid train_cfg in policy payload"

    missing = []
    if train_cfg.get("bet_penalty") is None:
        missing.append("bet_penalty")
    if train_cfg.get("ev_threshold") is None:
        missing.append("ev_threshold")

    use_obs_norm = bool(train_cfg.get("use_obs_norm", False))
    if use_obs_norm:
        obs_norm = policy.get("obs_norm")
        if not isinstance(obs_norm, dict) or "mean" not in obs_norm or "std" not in obs_norm:
            missing.append("obs_norm (mean/std)")

    if missing:
        reason = "missing required fields: " + ", ".join(missing)
        return False, reason

    return True, ""
