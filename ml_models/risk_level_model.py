"""scikit-learn classification model for traffic risk prediction."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml_models.common import DATA_DIR, MODEL_DIR, build_single_row_frame, ensure_model, save_pickle_model


MODEL_PATH = MODEL_DIR / "risk_level_model.pkl"
NUMERIC_FEATURES = [
    "hour",
    "traffic_volume",
    "air_pollution_index",
    "humidity",
    "visibility_in_miles",
    "temperature",
    "rain_p_h",
    "snow_p_h",
]
CATEGORICAL_FEATURES = ["weather_type", "weather_description"]
FEATURE_NAMES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _derive_weather_category(description: str) -> str:
    text = description.strip().lower()
    if re.search(r"thunder|storm", text):
        return "Storm"
    if re.search(r"snow|sleet", text):
        return "Snow"
    if re.search(r"fog|mist|haze|smoke", text):
        return "Low Visibility"
    if re.search(r"rain|drizzle", text):
        return "Rainy"
    if "cloud" in text:
        return "Cloudy"
    if "clear" in text:
        return "Clear"
    return "Other"


def _derive_risk_level(row: pd.Series) -> str:
    weather_category = _derive_weather_category(str(row["weather_description"]))
    weather_severity = {
        "Clear": 0,
        "Cloudy": 1,
        "Rainy": 2,
        "Low Visibility": 3,
        "Storm": 4,
        "Snow": 4,
        "Other": 1,
    }[weather_category]
    visibility_risk = 2 if row["visibility_in_miles"] < 2 else 1 if row["visibility_in_miles"] < 5 else 0
    pollution_risk = 2 if row["air_pollution_index"] > 150 else 1 if row["air_pollution_index"] > 100 else 0
    traffic_risk = 2 if row["traffic_volume"] > 6000 else 1 if row["traffic_volume"] > 3000 else 0
    total_score = weather_severity + visibility_risk + pollution_risk + traffic_risk
    if total_score >= 6:
        return "Severe"
    if total_score >= 4:
        return "High"
    if total_score >= 2:
        return "Moderate"
    return "Low"


def train_model() -> dict[str, Any]:
    """Train the traffic risk classification model and save it as a pickle file."""
    data_path = DATA_DIR / "traffic2.csv"
    df = pd.read_csv(data_path)
    df["date_time"] = pd.to_datetime(df["date_time"])
    df["hour"] = df["date_time"].dt.hour
    df["risk_level"] = df.apply(_derive_risk_level, axis=1)

    X = df[FEATURE_NAMES].copy()
    y = df["risk_level"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(n_estimators=200, random_state=42)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "f1_score": round(float(f1_score(y_test, predictions, average="weighted")), 4),
    }

    payload = {
        "model_name": "Traffic Risk Classifier",
        "model_type": "classification",
        "feature_names": FEATURE_NAMES,
        "target_name": "risk_level",
        "classes": sorted(y.unique().tolist()),
        "model": pipeline,
        "metrics": metrics,
    }
    save_pickle_model(payload, MODEL_PATH)
    return payload


def load_model() -> dict[str, Any]:
    """Load the traffic risk model, training it if needed."""
    return ensure_model(MODEL_PATH, train_model)


def predict(input_data: dict[str, Any]) -> dict[str, Any]:
    """Predict the traffic risk level using the common input payload."""
    payload = load_model()
    model = payload["model"]
    input_frame = build_single_row_frame(payload["feature_names"], input_data)
    prediction_label = str(model.predict(input_frame)[0])

    confidence_score = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_frame)[0]
        confidence_score = float(probabilities.max())

    return {
        "model_name": payload["model_name"],
        "prediction": prediction_label,
        "prediction_label": "Predicted risk level",
        "confidence_score": round(confidence_score, 4) if confidence_score is not None else None,
        "metrics": payload["metrics"],
    }

