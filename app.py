"""
app.py

Streamlit UI for predicting football correct scores.

Run:
  streamlit run app.py

Workflow:
  1) Train the model once:
       python train.py --matches data/spi_matches.csv --stadiums data/stadium_coordinates.csv --out models
  2) Start Streamlit:
       streamlit run app.py

The app will look for:
  models/score_models.joblib

"""

from __future__ import annotations

import importlib.util
import json
import os
from typing import Tuple

from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

from config import DEFAULT_ARTIFACT_ABS_PATH, DEFAULT_MATCHES_ABS_PATH, DEFAULT_STADIUMS_ABS_PATH
from features import (
    FeatureConfig,
    build_single_match_features,
    build_team_match_long,
    load_matches_csv,
    load_stadiums_csv,
)
from predict import (
    calibrate_scoreline_matrix,
    load_artifact,
    predict_expected_goals,
    scoreline_probability_matrix,
    top_scorelines,
)
from predict_helpers import prepare_and_validate_row


HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


def _resolve_matplotlib_cmap(cmap_name: str | None) -> str | None:
    """Resolve a colormap name to a matplotlib-registered cmap.

    - Returns a valid cmap name string when available.
    - Performs case-insensitive lookup and small alias handling.
    - Returns None only when matplotlib isn't available or name cannot be resolved.
    """
    if cmap_name is None:
        return None
    if not HAS_MATPLOTLIB:
        return None

    try:
        import matplotlib.pyplot as plt  # type: ignore

        name = str(cmap_name)
        # direct resolution
        try:
            plt.get_cmap(name)
            return name
        except Exception:
            pass

        # case-insensitive match against available colormaps
        cmap_list = list(plt.colormaps())
        name_lower = name.lower()
        for cm in cmap_list:
            if cm.lower() == name_lower:
                return cm

        # simple aliases for common user inputs
        aliases = {
            "ylorrd": "YlOrRd",
            "ylorr": "YlOrRd",
            "viridis": "viridis",
            "magma": "magma",
            "blues": "Blues",
        }
        if name_lower in aliases:
            candidate = aliases[name_lower]
            try:
                plt.get_cmap(candidate)
                return candidate
            except Exception:
                pass

        return None
    except Exception:
        return None


def _full_width_button(label: str, button_type: str = "secondary") -> bool:
    try:
        return st.button(label, type=button_type, width="stretch")
    except TypeError:
        try:
            return st.button(label, type=button_type, use_container_width=True)
        except TypeError:
            return st.button(label)


def _full_width_dataframe(data: Any, *, hide_index: bool | None = None) -> None:
    kwargs: dict[str, Any] = {}
    if hide_index is not None:
        kwargs["hide_index"] = hide_index

    try:
        st.dataframe(data, width="stretch", **kwargs)
        return
    except TypeError:
        pass

    try:
        st.dataframe(data, use_container_width=True, **kwargs)
        return
    except TypeError:
        pass

    st.dataframe(data, **kwargs)


if not HAS_MATPLOTLIB:
    st.error("matplotlib is required for colored Correct Score grid rendering. Install dependencies and rerun.")
    st.stop()


# ----------------------------
# Streamlit config
# ----------------------------

st.set_page_config(page_title="Football Correct Score Predictor", layout="wide")

st.markdown(
    """
<style>
div[data-testid="stMetric"] {
    border: 1px solid rgba(160, 160, 160, 0.25);
    border-radius: 10px;
    padding: 0.6rem 0.8rem;
    background-color: rgba(250, 250, 250, 0.02);
}

.app-hero {
    padding: 0.4rem 0 0.6rem 0;
}

.app-subtitle {
    color: #9aa0a6;
    margin-top: -0.2rem;
    margin-bottom: 0.8rem;
}

.section-title {
    font-size: 1.05rem;
    font-weight: 600;
    margin-top: 0.2rem;
    margin-bottom: 0.4rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------
# Caching helpers
# ----------------------------

@st.cache_resource(show_spinner=False)
def load_models(artifact_path: str) -> dict:
    return load_artifact(artifact_path)


@st.cache_data(show_spinner=True)
def load_data(matches_path: str, stadiums_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matches = load_matches_csv(matches_path)
    stadiums = load_stadiums_csv(stadiums_path) if stadiums_path else pd.DataFrame()
    long_df = build_team_match_long(matches)
    return matches, stadiums, long_df


def _league_team_lists(matches: pd.DataFrame) -> dict:
    leagues = sorted(matches["div"].dropna().unique().tolist())
    teams_by_league = {}
    for d in leagues:
        sub = matches[matches["div"] == d]
        teams = sorted(pd.unique(pd.concat([sub["home_team"], sub["away_team"]], axis=0)).tolist())
        teams_by_league[d] = teams
    return {"leagues": leagues, "teams_by_league": teams_by_league}


def _league_odds_medians(matches: pd.DataFrame) -> pd.DataFrame:
    # Useful defaults for UI (odds often missing for future matches)
    cols = ["div", "home_odds", "draw_odds", "away_odds"]
    df = matches[cols].copy()
    out = df.groupby("div")[["home_odds", "draw_odds", "away_odds"]].median(numeric_only=True).reset_index()
    return out


def _format_metric_label(metric_key: str) -> str:
    label = str(metric_key).replace("_", " ").title()
    replacements = {
        "Ece": "ECE",
        "Nll": "NLL",
        "Dc": "DC",
        "Dm": "DM",
        "Pi": "PI",
        "Wdl": "WDL",
    }
    for src, dst in replacements.items():
        label = label.replace(src, dst)
    return label


def _display_metric_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool, np.integer, np.floating, np.bool_)):
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (tuple, list, set, dict)):
        try:
            return json.dumps(value, default=str)
        except TypeError:
            return str(value)
    if pd.isna(value):
        return None
    return str(value)


# ----------------------------
# UI
# ----------------------------

st.markdown("<div class='app-hero'><h1>Football Correct Score Predictor</h1></div>", unsafe_allow_html=True)
st.markdown(
    "<p class='app-subtitle'>Estimate Expected Goals and full Correct Score probability distributions from historical match data.</p>",
    unsafe_allow_html=True,
)
st.divider()

default_matches_path = DEFAULT_MATCHES_ABS_PATH
default_stadiums_path = DEFAULT_STADIUMS_ABS_PATH
default_artifact_path = DEFAULT_ARTIFACT_ABS_PATH

with st.sidebar:
    st.header("Configuration")
    with st.expander("Advanced setup", expanded=False):
        st.markdown("**1) Model artifact**")
        artifact_path = st.text_input("Artifact path", value=default_artifact_path)

        st.markdown("**2) Input data**")
        matches_path = st.text_input("Matches CSV path", value=default_matches_path)
        stadiums_path = st.text_input("Stadiums CSV path", value=default_stadiums_path)

        st.info(
            "If you haven't trained yet, open a terminal in this folder and run:\n\n"
            "`python train.py --matches data/spi_matches.csv --stadiums data/stadium_coordinates.csv --out models`"
        )

    with st.expander("Quick guide", expanded=False):
        st.markdown(
            """
1. **Select match inputs**
   - Pick league, home team, away team, and match date.
    - Enter market odds (recommended for stronger pre-match signal).

2. **Configure Dixon-Coles (optional)**
   - Enable **Use Dixon-Coles low-score correction** to adjust low-score probabilities.
    - Use the rho slider to test sensitivity.

3. **Run prediction**
    - Click **Run prediction**.
    - Review Expected Goals (`λ_home`, `λ_away`) and 1X2 probabilities.

4. **Interpret outputs**
    - **Top Correct Scores** shows the highest-probability exact scores.
    - **Correct Score probability grid** shows the full distribution (higher color intensity = higher probability).
   - If enabled, **Poisson vs Dixon-Coles** helps compare baseline vs corrected probabilities.

5. **Practical tip**
    - Use the full probability distribution, not only the top prediction.
            """
        )

    st.subheader("Diagnostics")
    concentration_threshold = st.slider(
          "Top Correct Score concentration alert threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
          help="Show an alert when the most common top predicted Correct Score exceeds this share on the test split.",
    )

# Defaults used when advanced setup inputs are not opened by user
if "artifact_path" not in locals():
    artifact_path = default_artifact_path
if "matches_path" not in locals():
    matches_path = default_matches_path
if "stadiums_path" not in locals():
    stadiums_path = default_stadiums_path

# If artifact missing, optionally download from MODEL_ARTIFACT_URL (set as an env var)
def _maybe_fetch_remote_artifact(artifact_path: str) -> bool:
    url = os.environ.get("MODEL_ARTIFACT_URL") or os.environ.get("MODEL_URL")
    if not url:
        return False
    try:
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        p = Path(artifact_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    fh.write(chunk)
        return True
    except Exception as e:
        # safe to continue - will be handled by artifact_ok check below
        st.warning(f"Failed to download model artifact from {url}: {e}")
        return False

# Load model artifact (try download-if-missing first)
if not Path(artifact_path).exists():
    if _maybe_fetch_remote_artifact(artifact_path):
        st.info("Downloaded model artifact from MODEL_ARTIFACT_URL")

# Load model artifact
artifact_ok = Path(artifact_path).exists()
if not artifact_ok:
    st.error(f"Model artifact not found at: {artifact_path}")
    st.stop()

try:
    artifact = load_models(artifact_path)
except Exception as exc:
    msg = str(exc)
    st.error("Unable to load model artifact.")
    st.exception(exc)
    if "_RemainderColsList" in msg or "sklearn" in msg.lower() or "unpickle" in msg.lower():
        st.info(
            "This usually means a scikit-learn version mismatch between the saved artifact and your active environment.\n\n"
            "Try one of these fixes:\n"
            "1) Install the training-compatible version: `pip install scikit-learn==1.6.1`\n"
            "2) Or retrain in your current environment to regenerate `models/score_models.joblib`."
        )
    st.stop()

cfg_raw = artifact.get("config", {})
time_decay_cfg = artifact.get("time_decay", {})
decay_tuning_cfg = artifact.get("decay_tuning", {})
diagnostics_cfg = artifact.get("diagnostics", {})
top_score_freq_cfg = diagnostics_cfg.get("top_score_frequency_test", {})
benchmark_cfg = diagnostics_cfg.get("benchmark_panel", {})
dm_cfg = diagnostics_cfg.get("diebold_mariano", {}).get("model_vs_empirical", {})
lambda_interval_cfg = diagnostics_cfg.get("lambda_interval", {})
rolling_bt_cfg = diagnostics_cfg.get("rolling_backtest", {})
scientific_mode_cfg = artifact.get("scientific_mode", {})
scientific_mode_criteria = scientific_mode_cfg.get("criteria", {})
scoreline_calibration_cfg = artifact.get("scoreline_calibration", {})
top1_reliability_cfg = scoreline_calibration_cfg.get("top1_reliability", {})
event_reliability_cfg = scoreline_calibration_cfg.get("event_reliability", {})
event_reliability_events = event_reliability_cfg.get("events", {}) if isinstance(event_reliability_cfg, dict) else {}
dixon_coles_rho = artifact.get("dixon_coles_rho")
dixon_coles_cfg = artifact.get("dixon_coles", {})
dixon_coles_global = dixon_coles_cfg.get("rho_global", dixon_coles_rho)
dixon_coles_by_div = dixon_coles_cfg.get("rho_by_div", {}) or {}
rho_bounds = dixon_coles_cfg.get("rho_search_bounds", {"min": -0.30, "max": 0.30})
rmin = float(rho_bounds.get("min", -0.30))
rmax = float(rho_bounds.get("max", 0.30))
config = FeatureConfig(
    windows=tuple(cfg_raw.get("windows", [5, 10])),
    max_goals=int(cfg_raw.get("max_goals", 6)),
    use_travel_distance=bool(cfg_raw.get("use_travel_distance", True)),
    ewm_span=int(cfg_raw.get("ewm_span", 6)),
    use_ewm_features=bool(cfg_raw.get("use_ewm_features", True)),
    use_adjusted_features=bool(cfg_raw.get("use_adjusted_features", True)),
)

with st.sidebar:
    with st.expander("Scientific diagnostics", expanded=False):
        diag_items = {
            "strict_mode": scientific_mode_cfg.get("strict_scientific_mode", False),
            "benchmark_model_top1": benchmark_cfg.get("model_top1_acc"),
            "benchmark_always_1_1": benchmark_cfg.get("always_1_1_acc"),
            "benchmark_uniform_random": benchmark_cfg.get("uniform_random_acc"),
            "dm_model_vs_empirical_p": dm_cfg.get("p_value"),
            "lambda_pi_home_coverage": (lambda_interval_cfg.get("home") or {}).get("coverage"),
            "lambda_pi_away_coverage": (lambda_interval_cfg.get("away") or {}).get("coverage"),
            "rolling_backtest_folds": rolling_bt_cfg.get("n_folds"),
            "rolling_backtest_nll_mean": rolling_bt_cfg.get("nll_mean"),
            "rolling_backtest_nll_std": rolling_bt_cfg.get("nll_std"),
        }
        st.write(diag_items)

# Load data
try:
    matches, stadiums, long_df = load_data(matches_path, stadiums_path)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

lists = _league_team_lists(matches)
leagues = lists["leagues"]
teams_by_league = lists["teams_by_league"]
odds_medians = _league_odds_medians(matches)

max_date = matches["date"].max()
default_match_date = (max_date + pd.Timedelta(days=1)).date() if pd.notna(max_date) else pd.Timestamp.today().date()

colA, colB = st.columns([1, 1], gap="large")

with colA:
    st.markdown("<div class='section-title'>Match Inputs</div>", unsafe_allow_html=True)

    div = st.selectbox("League (div)", leagues, index=0 if leagues else None)
    teams = teams_by_league.get(div, [])

    home_team = st.selectbox("Home team", teams, index=0 if teams else None)
    away_team = st.selectbox("Away team", teams, index=1 if len(teams) > 1 else 0)

    match_date = st.date_input("Match date (as-of)", value=default_match_date)
    asof_ts = pd.Timestamp(match_date)

    med_row = odds_medians[odds_medians["div"] == div]
    if len(med_row) == 1:
        default_home_odds = float(med_row["home_odds"].iloc[0]) if pd.notna(med_row["home_odds"].iloc[0]) else 2.2
        default_draw_odds = float(med_row["draw_odds"].iloc[0]) if pd.notna(med_row["draw_odds"].iloc[0]) else 3.3
        default_away_odds = float(med_row["away_odds"].iloc[0]) if pd.notna(med_row["away_odds"].iloc[0]) else 3.0
    else:
        default_home_odds, default_draw_odds, default_away_odds = 2.2, 3.3, 3.0

    st.markdown("<div class='section-title'>Market Odds</div>", unsafe_allow_html=True)
    home_odds = st.number_input("Home odds", min_value=1.01, value=float(default_home_odds), step=0.01)
    draw_odds = st.number_input("Draw odds", min_value=1.01, value=float(default_draw_odds), step=0.01)
    away_odds = st.number_input("Away odds", min_value=1.01, value=float(default_away_odds), step=0.01)

    league_rho_default = dixon_coles_by_div.get(str(div)) if div is not None else None
    if league_rho_default is None:
        league_rho_default = dixon_coles_global

    dc_available = league_rho_default is not None
    use_dixon_coles = st.checkbox(
        "Use Dixon-Coles low-score correction",
        value=bool(dc_available),
        disabled=not dc_available,
    )
    if dc_available:
        rho_selected = st.slider(
            "Dixon-Coles rho",
            min_value=float(rmin),
            max_value=float(rmax),
            value=float(league_rho_default),
            step=0.001,
        )
    else:
        rho_selected = None
        st.caption("Train with `--fit-dc` to enable Dixon-Coles controls.")

    st.markdown("<div class='section-title'>Prediction Controls</div>", unsafe_allow_html=True)
    predict_btn = _full_width_button("Run prediction", button_type="primary")
    compare_poisson_dc = st.checkbox("Compare baseline (Poisson) vs Dixon-Coles", value=False)
    grid_color_scale = st.selectbox(
        "Grid color scale",
        ["YlOrRd", "Blues", "Greens", "Viridis", "Magma"],
        index=0,
    )
    grid_cmap_lookup = {
        "YlOrRd": "YlOrRd",
        "Blues": "Blues",
        "Greens": "Greens",
        "Viridis": "viridis",
        "Magma": "magma",
    }


with colB:
    st.markdown("<div class='section-title'>Model Info</div>", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Train matches", f"{int(artifact.get('train_rows') or 0):,}")
    m2.metric("Test matches", f"{int(artifact.get('test_rows') or 0):,}")
    m3.metric("Grid max goals", f"{int(config.max_goals)}")

    model_info_items = {
        "windows": config.windows,
        "use_travel_distance": config.use_travel_distance,
        "ewm_span": config.ewm_span,
        "use_ewm_features": config.use_ewm_features,
        "use_adjusted_features": config.use_adjusted_features,
        "cutoff_date": artifact.get("cutoff_date"),
        "data_max_date": artifact.get("max_date"),
        "time_decay_half_life_days": time_decay_cfg.get("half_life_days"),
        "time_decay_ref_date_train_max": time_decay_cfg.get("ref_date_train_max"),
        "time_decay_min_weight": time_decay_cfg.get("min_weight"),
        "decay_tuning_enabled": decay_tuning_cfg.get("enabled", False),
        "decay_tuning_metric": decay_tuning_cfg.get("metric"),
        "decay_tuning_selected_half_life_days": decay_tuning_cfg.get("selected_half_life_days"),
        "decay_tuning_val_days": decay_tuning_cfg.get("val_days"),
        "strict_scientific_mode": scientific_mode_cfg.get("strict_scientific_mode", False),
        "dc_significance_z": scientific_mode_criteria.get("dc_significance_z"),
        "dc_max_top_share": scientific_mode_criteria.get("dc_max_top_share"),
        "scoreline_calibration_enabled": scoreline_calibration_cfg.get("enabled", False),
        "scoreline_calibration_method": scoreline_calibration_cfg.get("method"),
        "scoreline_calibration_temperature": scoreline_calibration_cfg.get("temperature"),
        "scoreline_calibration_by_league": bool(scoreline_calibration_cfg.get("temperatures_by_div")),
        "scoreline_calibration_nll_improvement": scoreline_calibration_cfg.get("nll_improvement"),
        "scoreline_calibration_top1_ece": top1_reliability_cfg.get("ece"),
        "low_score_mixture_enabled": (scoreline_calibration_cfg.get("low_score_mixture") or {}).get("enabled"),
        "low_score_mixture_alpha": (scoreline_calibration_cfg.get("low_score_mixture") or {}).get("alpha"),
        "low_score_mixture_nll_improvement": (scoreline_calibration_cfg.get("low_score_mixture") or {}).get("nll_improvement"),
        "dixon_coles_rho_source": dixon_coles_cfg.get("rho_global_source"),
        "dixon_coles_rho_global": dixon_coles_global,
        "dixon_coles_rho_league": dixon_coles_by_div.get(str(div)) if div else None,
        "benchmark_model_top1": benchmark_cfg.get("model_top1_acc"),
        "benchmark_always_1_1": benchmark_cfg.get("always_1_1_acc"),
        "benchmark_uniform_random": benchmark_cfg.get("uniform_random_acc"),
        "dm_model_vs_empirical_p": dm_cfg.get("p_value"),
        "lambda_pi_home_coverage": (lambda_interval_cfg.get("home") or {}).get("coverage"),
        "lambda_pi_away_coverage": (lambda_interval_cfg.get("away") or {}).get("coverage"),
        "rolling_backtest_folds": rolling_bt_cfg.get("n_folds"),
        "rolling_backtest_nll_mean": rolling_bt_cfg.get("nll_mean"),
    }

    info_df = pd.DataFrame(
        [
            {
                "metric": _format_metric_label(key),
                "value": _display_metric_value(value),
            }
            for key, value in model_info_items.items()
        ]
    )
    info_df["metric"] = info_df["metric"].astype(str)
    info_df["value"] = info_df["value"].map(lambda x: "" if x is None else str(x))
    _full_width_dataframe(info_df, hide_index=True)

    rows = top_score_freq_cfg.get("rows", []) if isinstance(top_score_freq_cfg, dict) else []
    if rows:
        st.caption("Top predicted Correct Score frequencies on test split")
        top_row = rows[0]
        top_share = float(top_row.get("share", 0.0))
        if top_share > concentration_threshold:
            st.warning(
                f"Concentration alert: {top_row.get('score')} is the top predicted Correct Score in {top_share:.1%} of test matches."
            )
        diag_df = pd.DataFrame(rows)
        if "share" in diag_df.columns:
            diag_df["share"] = (diag_df["share"] * 100.0).map(lambda x: f"{x:.2f}%")
        _full_width_dataframe(diag_df, hide_index=True)

    if event_reliability_events:
        st.caption("Correct Score event reliability (validation split during training)")

        cue_green_max = 0.01   # 1.0 pp
        cue_yellow_max = 0.03  # 3.0 pp

        def _cue_from_error(err: float) -> str:
            if err <= cue_green_max:
                return "Good"
            if err <= cue_yellow_max:
                return "Watch"
            return "Alert"

        summary_rows = []
        for event in ("1-1", "1-0", "0-1", "0-0"):
            info = event_reliability_events.get(event)
            if not info:
                continue
            mean_pred = float(info.get("mean_pred", np.nan))
            prevalence = float(info.get("prevalence", np.nan))
            ece = float(info.get("ece", np.nan))
            abs_gap = float(abs(mean_pred - prevalence))
            severity = max(abs_gap, ece)
            summary_rows.append(
                {
                    "cue": _cue_from_error(severity),
                    "event": event,
                    "ece": ece,
                    "brier": float(info.get("brier", np.nan)),
                    "abs_gap": abs_gap,
                    "mean_pred": mean_pred,
                    "prevalence": prevalence,
                }
            )
        if summary_rows:
            ev_df = pd.DataFrame(summary_rows)
            ev_df["ece"] = (ev_df["ece"] * 100.0).map(lambda x: f"{x:.2f}%")
            ev_df["abs_gap"] = (ev_df["abs_gap"] * 100.0).map(lambda x: f"{x:.2f} pp")
            for col in ["mean_pred", "prevalence"]:
                ev_df[col] = (ev_df[col] * 100.0).map(lambda x: f"{x:.2f}%")
            _full_width_dataframe(ev_df, hide_index=True)

            with st.expander("Show event reliability bins"):
                for event in ("1-1", "1-0", "0-1", "0-0"):
                    info = event_reliability_events.get(event)
                    if not info:
                        continue
                    st.markdown(f"**{event}**")
                    bins = pd.DataFrame(info.get("rows", []))
                    if len(bins) == 0:
                        st.caption("No reliability bins available.")
                    else:
                        bins["cue"] = bins["gap"].abs().map(lambda x: _cue_from_error(float(x)))
                        bins["mean_pred"] = (bins["mean_pred"] * 100.0).map(lambda x: f"{x:.2f}%")
                        bins["empirical_rate"] = (bins["empirical_rate"] * 100.0).map(lambda x: f"{x:.2f}%")
                        bins["gap"] = (bins["gap"] * 100.0).map(lambda x: f"{x:+.2f} pp")
                        _full_width_dataframe(bins, hide_index=True)

    st.caption("Note: Correct Score prediction is inherently uncertain; rely on the distribution, not only the top pick.")

if predict_btn:
    if not div or not home_team or not away_team:
        st.warning("Select a league, a home team, and an away team before running prediction.")
        st.stop()

    if home_team == away_team:
        st.warning("Home team and away team must be different.")
        st.stop()

    # Build single-row feature frame
    feat_row = build_single_match_features(
        div=div,
        home_team=home_team,
        away_team=away_team,
        asof_date=asof_ts,
        home_odds=float(home_odds),
        draw_odds=float(draw_odds),
        away_odds=float(away_odds),
        long_df=long_df,
        stadiums=stadiums,
        config=config,
    )

    # Validate and prepare model inputs (assert column parity with artifact)
    try:
        X = prepare_and_validate_row(
            artifact=artifact,
            matches=matches,
            stadiums=stadiums,
            div=div,
            home_team=home_team,
            away_team=away_team,
            asof_date=asof_ts,
            home_odds=float(home_odds),
            draw_odds=float(draw_odds),
            away_odds=float(away_odds),
        )
    except Exception as e:
        st.error(f"Preprocessing validation failed: {e}")
        st.stop()

    # Runtime assertion: ensure columns exactly match artifact expectation
    expected_cols = list(artifact.get("cat_cols", []) + artifact.get("num_cols", []))
    try:
        assert list(X.columns) == expected_cols
    except AssertionError:
        st.error("Model input columns do not match the artifact's expected columns.")
        st.stop()

    # Friendly warning: show any features that will be imputed by the pipeline
    imputed_cols = [c for c in X.columns if pd.isna(feat_row.at[0, c])]
    if imputed_cols:
        st.warning(
            "Some input features are missing and will be imputed by the model: " +
            ", ".join(imputed_cols)
        )

        # Short explanatory note for users about what imputation means
        st.info(
            "Missing features (for example travel distance or recent form) are filled with "
            "default values by the model pipeline. This reduces personalization and may "
            "degrade prediction accuracy for that match. Consider providing odds and "
            "ensuring teams exist in the historical dataset for best results."
        )

        # Optional telemetry: POST a non-blocking imputation event if a telemetry URL is set
        telemetry_url = os.environ.get("IMPUTATION_TELEMETRY_URL") or os.environ.get("MODEL_TELEMETRY_URL")
        if telemetry_url:
            try:
                payload = {
                    "artifact": str(artifact_path) if "artifact_path" in globals() else None,
                    "cutoff_date": artifact.get("cutoff_date"),
                    "div": div,
                    "home_team": home_team,
                    "away_team": away_team,
                    "imputed_cols": imputed_cols,
                    "imputed_count": len(imputed_cols),
                }
                # Best-effort POST; failures should not affect UX
                try:
                    requests.post(telemetry_url, json=payload, timeout=2.0)
                except Exception:
                    pass
            except Exception:
                pass

        # Also append a local timestamped log (safe best-effort)
        try:
            import pathlib, datetime

            log_dir = pathlib.Path("logs")
            log_dir.mkdir(exist_ok=True)
            log_path = log_dir / "imputation_events.log"
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"{datetime.datetime.utcnow().isoformat()}Z\t{div}\t{home_team}\t{away_team}\t{','.join(imputed_cols)}\n")
        except Exception:
            pass

    lam_home, lam_away = predict_expected_goals(artifact, X)
    lam_h = float(lam_home[0])
    lam_a = float(lam_away[0])

    st.divider()
    st.markdown("<div class='section-title'>Prediction Results</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Expected home goals (λ_home)", f"{lam_h:.2f}")
    c2.metric("Expected away goals (λ_away)", f"{lam_a:.2f}")

    active_rho = float(rho_selected) if (use_dixon_coles and rho_selected is not None) else None

    mat = scoreline_probability_matrix(
        lam_h,
        lam_a,
        max_goals=config.max_goals,
        include_tail_bucket=True,
        rho=active_rho,
    )
    mat = calibrate_scoreline_matrix(mat, scoreline_calibration_cfg, div=str(div) if div else None)

    arr_main = mat.to_numpy(dtype=float)
    wdl = {
        "home_win": float(np.tril(arr_main, k=-1).sum()),
        "draw": float(np.trace(arr_main)),
        "away_win": float(np.triu(arr_main, k=1).sum()),
    }
    c3.metric("Home / Draw / Away", f"{wdl['home_win']:.2%} / {wdl['draw']:.2%} / {wdl['away_win']:.2%}")

    wdl_df = pd.DataFrame(
        {
            "Probability": [wdl["home_win"], wdl["draw"], wdl["away_win"]],
        },
        index=["Home Win", "Draw", "Away Win"],
    )
    st.markdown("### 1X2 Outcome Probabilities")
    st.bar_chart(wdl_df)

    top = top_scorelines(mat, top_n=10)
    top_poisson = None

    if active_rho is not None:
        mat_poisson = scoreline_probability_matrix(
            lam_h,
            lam_a,
            max_goals=config.max_goals,
            include_tail_bucket=True,
            rho=None,
        )
        mat_poisson = calibrate_scoreline_matrix(mat_poisson, scoreline_calibration_cfg, div=str(div) if div else None)
        top_poisson = top_scorelines(mat_poisson, top_n=10)

        poisson_score = str(top_poisson.iloc[0]["score"])
        dc_score = str(top.iloc[0]["score"])
        poisson_prob = float(top_poisson.iloc[0]["prob"])
        dc_prob = float(top.iloc[0]["prob"])
        delta_pp = (dc_prob - poisson_prob) * 100.0
        sign = "+" if delta_pp >= 0 else ""
        delta_color = "green" if delta_pp >= 0 else "red"

        if poisson_score == dc_score:
            message = f"Top Correct Score {dc_score} changed by {sign}{delta_pp:.2f} pp (Poisson → Dixon-Coles)."
            if delta_pp >= 0:
                st.success(message)
            else:
                st.warning(message)
        else:
            st.caption(
                f"Top Correct Score changed: Poisson `{poisson_score}` ({poisson_prob:.2%}) → Dixon-Coles `{dc_score}` ({dc_prob:.2%})."
            )

    tab_overview, tab_top, tab_grid, tab_debug = st.tabs([
        "Overview",
        "Top Correct Scores",
        "Correct Score Grid",
        "Feature Debug",
    ])

    with tab_overview:
        lead = top.iloc[0]
        st.success(f"Most likely Correct Score: {lead['score']} ({float(lead['prob']):.2%})")
        _full_width_dataframe(top.head(5).style.format({"prob": "{:.2%}"}))

    with tab_top:
        if compare_poisson_dc and active_rho is not None:
            top_dc = top
            cc1, cc2 = st.columns(2)
            with cc1:
                st.caption("Poisson")
                _full_width_dataframe(top_poisson.style.format({"prob": "{:.2%}"}))
            with cc2:
                st.caption("Dixon-Coles")
                _full_width_dataframe(top_dc.style.format({"prob": "{:.2%}"}))
        else:
            if compare_poisson_dc and active_rho is None:
                st.info("No `dixon_coles_rho` value found in artifact; showing baseline output.")
            _full_width_dataframe(top.style.format({"prob": "{:.2%}"}))

    with tab_grid:
        grid_styler = mat.style.format("{:.2%}")
        selected_cmap = _resolve_matplotlib_cmap(grid_cmap_lookup.get(grid_color_scale))
        if selected_cmap is None:
            # robust fallback: don't crash the app if a colormap name is unrecognized.
            # pick a visually-meaningful default and surface a warning to the user.
            fallback = "viridis"
            try:
                st.warning(f"Colormap '{grid_color_scale}' not found in Matplotlib; using '{fallback}'.")
            except Exception:
                pass
            selected_cmap = fallback
        grid_styler = grid_styler.background_gradient(cmap=selected_cmap, axis=None)
        _full_width_dataframe(grid_styler)

    with tab_debug:
        _full_width_dataframe(feat_row)
