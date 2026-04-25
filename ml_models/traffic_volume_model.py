"""scikit-learn regression model for predicting vehicle count."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from ml_models.common import DATA_DIR, MODEL_DIR, build_single_row_frame, ensure_model, save_pickle_model


MODEL_PATH = MODEL_DIR / "traffic_volume_model.pkl"
FEATURE_NAMES = ["hour", "junction"]


def train_model() -> dict[str, Any]:
    """Train the traffic volume regression model and save it as a pickle file."""
    data_path = DATA_DIR / "traffic1.csv"
    df = pd.read_csv(data_path)
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df["hour"] = df["DateTime"].dt.hour
    df = df.rename(columns={"Junction": "junction"})

    X = df[FEATURE_NAMES]
    y = df["Vehicles"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {
        "mae": round(float(mean_absolute_error(y_test, predictions)), 2),
        "r2": round(float(r2_score(y_test, predictions)), 4),
    }

    payload = {
        "model_name": "Traffic Volume Regressor",
        "model_type": "regression",
        "feature_names": FEATURE_NAMES,
        "target_name": "Vehicles",
        "model": model,
        "metrics": metrics,
    }
    save_pickle_model(payload, MODEL_PATH)
    return payload


def load_model() -> dict[str, Any]:
    """Load the traffic volume model, training it if the file does not exist yet."""
    return ensure_model(MODEL_PATH, train_model)


def predict(input_data: dict[str, Any]) -> dict[str, Any]:
    """Predict vehicle count using the common input payload."""
    payload = load_model()
    model = payload["model"]
    input_frame = build_single_row_frame(payload["feature_names"], input_data)
    prediction_value = float(model.predict(input_frame)[0])

    input_array = input_frame.to_numpy()
    tree_predictions = np.array([tree.predict(input_array)[0] for tree in model.estimators_])
    spread = float(np.std(tree_predictions))
    confidence_score = max(0.0, min(1.0, 1.0 - (spread / max(abs(prediction_value), 1.0))))

    return {
        "model_name": payload["model_name"],
        "prediction": round(prediction_value, 2),
        "prediction_label": "Predicted vehicle count",
        "confidence_score": round(confidence_score, 4),
        "metrics": payload["metrics"],
    }
