# Feature Engineering Analysis & Critical Issues

## Q1: Are features computed longitudinally or just baseline?

### ✅ EXISTING Features (Working Well):
- **Rolling form features** (`add_rolling_form_features`, lines 127-168):
  - Uses `.rolling(window=w, min_periods=1).mean()` → computes over 5, 10-match windows
  - **Correctly shifted**: `.shift(1)` to avoid look-ahead bias ✓
  - EWM features use `.ewm(span=span).mean().shift(1)` → exponential decay ✓
  - **Temporal structure**: LONGITUDINAL (cumulative up to previous match)

### ❌ NEW Features (INTRODUCED BUGS):

**Problem**: My new features have TEMPORAL LEAKAGE:

1. **team_strength_features** (lines 228-262):
   ```python
   long_df["team_wins_cumulative"] = g["points"].transform(lambda s: (s > 1).astype(int).cumsum().shift(1))  # ✓ Shifted
   long_df["team_matches_cumulative"] = g.cumcount()  # ❌ NOT shifted - looks forward!
   long_df["team_win_pct"] = long_df["team_wins_cumulative"] / long_df["team_matches_cumulative"]
   ```
   - **LEAK**: `cumcount()` gives match 0,1,2,3... but we're using it as denominator for match 0,1,2,3 (future-looking)
   - Should be: `g.cumcount().shift(1)` 

2. **league_form_aggregates** (lines 266-285):
   ```python
   group["league_gf_cumean"] = group["goals_for"].expanding().mean()
   # Then later: group["league_gf_cumean"] = group.groupby(["div"])["league_gf_cumean"].shift(1)
   ```
   - **PROBLEM**: The expanding() happens per-division, but then we shift - however this is applied OUTSIDE of match-by-match context
   - Not tied to team history, just league history. Less problematic but poorly structured.

3. **rest_variance_patterns** (lines 289-308):
   - Uses `groupby().apply()` which can break temporal ordering
   - Cutting rest_days into buckets on small samples is unreliable
   - Not actually longitudinal - just a static variance metric

4. **home_advantage_variance** (lines 312-330):
   - Manually iterates and assigns values with `.loc[]` in groups
   - Loses temporal structure entirely
   - Computes from ALL historical data (cumulative), not windowed

## Q2: Is data correctly structured for temporal/time-series format?

### ✓ YES, but only if features respect it:
```
data/spi_matches.csv structure:
- date: YYYY-MM-DD (2019-08-01 through 2026-03-02)
- div: League identifier (33 leagues)
- home_team, away_team: Team names
- Sorted by [date, div, home_team, away_team]
```

### ✓ build_team_match_long() correctly restructures:
```
Original: Match (home, away, score)
Long:     Team A perspective (home, opponent, points, is_home=1)
         Team A perspective (away, opponent, points, is_home=0)
         → Sorted by [div, team, date]
```

### ❌ My new features don't respect this:
- They compute characteristics but don't properly scope them to "what we know at prediction time"
- Example: `team_home_advantage` uses ALL historical data, not windowed by recency

## Q3: How much historical data required?

### Current approach: **Unlimited lookback with decay**
```python
# build_time_decay_weights() - train.py lines 65-84
weights = 0.5 ** (delta_days / half_life_days)

# Current training output shows:
# "[Tuning] Using fixed half-life=0.0 days"
# Time-decay: disabled (equal weighting)
```

### Data availability:
- **Training data**: 168,920 matches (2019-08-01 to 2025-03-02)
- **Test data**: 12,362 matches (2025-03-02 to 2026-03-02)
- **Effective seasons**: ~6 full seasons of data
- **Teams with full history**: ~400+ teams across 33 leagues

### My new features problem:
- **team_strength_features**: Uses entire career (ALL seasons)
  - Team promoted mid-season? Uses data from lower league ❌
  - Team changed ownership/coach? Still uses old data ❌
  - Better: Limit to last 2-3 seasons or use rolling decay

- **league_form_aggregates**: Uses entire season up to date
  - League evolves (teams promoted/relegated)
  - Better: Use windowed (e.g., last 50 league matches)

## Q4: Does model use time-decay?

### ✓ YES, but it's DISABLED by default

From train.py line 974-976:
```python
w = build_time_decay_weights(
    train_df["date"],
    cutoff_date=args.cutoff_date,
    half_life_days=half_life_days,
)
```

Current training output shows:
```
[Tuning] Using fixed half-life=0.0 days
Time-decay: disabled (equal weighting)
```

### Why disabled?
- Default `--half-life-days: 0` (no time decay parameter in CLI args shown)
- Training treats all historical matches equally
- Recent matches weighted same as 6-year-old matches

### Impact on concentration:
- **OLD matches** (2019-2021) had different pace, scoring patterns
- **Averaging 6 years** of data causes features to be bland/averaged
- This is a ROOT CAUSE of 1-1 concentration!

## Q5: Is RL policy important? How to enhance?

### Current RL policy:
- Location: `models/rl_policy.joblib` 
- Type: Policy trained via reinforcement learning (Q-learning or policy gradient)
- Purpose: Learn optimal bet sizing given model predictions
- Status: Now in git (commit 897ec9f)

### Relevance to 1-1 concentration:
- **RL policy is DOWNSTREAM** of scoreline predictions
- If base model predicts only 1-1, RL policy can't learn variability
- **NOT the root cause** - model prediction variance is the issue

### Ways to enhance RL policy:
1. **Retrain after feature improvements**: If base model predicts diverse outcomes, RL learns better
2. **Use ensemble**: Train on multiple model checkpoints (early, mid, late training)
3. **Add exploration**: Current policy might be overfitting to training bets
4. **Add implicit trading costs**: Fee per bet changes optimal policy
5. **Use double Q-learning**: Reduce overestimation bias

---

## CRITICAL FINDINGS

### Why did 1-1 concentration WORSEN (62.3% → 64.28%)?

**Root causes**:
1. **Temporal leakage** in team_strength_features and home_advantage_variance
2. **League aggregates** are too coarse (all teams lumped together)
3. **Rest variance** computed on too-small sample sizes
4. **No time decay** - 6-year average smooths out all signal
5. **Features not additive to rolling form** - they're replacing structure, not enhancing it

### Why same issue before? (62.3% on previous training)

The 62% score from the previous training (before my feature additions) was already the baseline issue:
- Low feature variance due to limited diff signals
- Implied odds don't differentiate well
- No time-decay enabled
- Teams with 6 years of history still weighted equally

### What would actually help:

1. **Enable time-decay**: `--half-life-days 730` (2 years)
   - Recent performance weighted more
   - Reduces stale historical data influence
   
2. **Proper team strength**: Rolling form windows ALREADY exist
   - `gf_mean_5`, `gf_mean_10` already capture recency
   - No need for cumulative strength

3. **Feature engineering focus**:
   - Interaction features (rest × opponent_strength)
   - Momentum: (recent form) - (old form)
   - Player availability / injury status (not in data)
   - League-position-based pricing (teams higher in table vs lower)

4. **Structural fixes**:
   - Separate models per league (33 leagues have different pace)
   - Separate models by season (2019-2025 scoring evolved)
   - Non-linear feature relationships (splines for home advantage)
