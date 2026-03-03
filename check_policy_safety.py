import joblib
from rl_policy_utils import policy_is_safe

policy = joblib.load('models/rl_policy.joblib')
ok, reason = policy_is_safe(policy)

print(f"Safe: {ok}")
print(f"Reason: '{reason}'")
print(f"\nPolicy structure:")
print(f"  - Has W: {('W' in policy)}")
print(f"  - Has b: {('b' in policy)}")
print(f"  - Has train_cfg: {('train_cfg' in policy)}")
print(f"  - Has obs_norm: {('obs_norm' in policy)}")

if 'train_cfg' in policy:
    cfg = policy['train_cfg']
    print(f"\nTraining config:")
    print(f"  - bet_penalty: {cfg.get('bet_penalty')}")
    print(f"  - ev_threshold: {cfg.get('ev_threshold')}")
    print(f"  - use_obs_norm: {cfg.get('use_obs_norm')}")

if 'obs_norm' in policy:
    print(f"\nObservation normalization:")
    print(f"  - Has mean: {('mean' in policy['obs_norm'])}")
    print(f"  - Has std: {('std' in policy['obs_norm'])}")
