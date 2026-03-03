from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_MATCHES_REL_PATH = "data/spi_matches.csv"
DEFAULT_STADIUMS_REL_PATH = "data/stadium_coordinates_completed_full.csv"
DEFAULT_MODELS_DIR = "models"
DEFAULT_ARTIFACT_FILENAME = "score_models.joblib"


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


DEFAULT_MATCHES_ABS_PATH = str(project_path("data", "spi_matches.csv"))
DEFAULT_STADIUMS_ABS_PATH = str(project_path("data", "stadium_coordinates_completed_full.csv"))
DEFAULT_ARTIFACT_ABS_PATH = str(project_path("models", DEFAULT_ARTIFACT_FILENAME))
