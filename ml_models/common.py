"""Shared helpers for ML training, loading, validation, and prediction."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


def save_pickle_model(payload: dict[str, Any], model_path: Path) -> None:
    """Save a trained model payload to disk."""
    with model_path.open("wb") as model_file:
        pickle.dump(payload, model_file)


def load_pickle_model(model_path: Path) -> dict[str, Any]:
    """Load a previously saved model payload."""
    with model_path.open("rb") as model_file:
        return pickle.load(model_file)


def ensure_model(model_path: Path, trainer) -> dict[str, Any]:
    """Train the model if needed, then return the saved payload."""
    if not model_path.exists():
        trainer()
    return load_pickle_model(model_path)


def build_single_row_frame(feature_names: list[str], input_data: dict[str, Any]) -> pd.DataFrame:
    """Create a single-row DataFrame in the required feature order."""
    return pd.DataFrame([[input_data[name] for name in feature_names]], columns=feature_names)

