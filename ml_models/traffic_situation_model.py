"""scikit-learn classification model for traffic situation prediction."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml_models.common import DATA_DIR, MODEL_DIR, build_single_row_frame, ensure_model, save_pickle_model


MODEL_PATH = MODEL_DIR / "traffic_situation_model.pkl"
FEATURE_NAMES = ["car_count", "bike_count", "bus_count", "truck_count", "traffic_volume", "day_of_week", "hour"]
CATEGORICAL_FEATURES = ["day_of_week"]
NUMERIC_FEATURES = ["car_count", "bike_count", "bus_count", "truck_count", "traffic_volume", "hour"]


def train_model() -> dict:
    """Train the traffic situation classification model and save it as a pickle file."""
    data_path = DATA_DIR / "traffic3.csv"
    df = pd.read_csv(data_path)
    df["Time"] = pd.to_datetime(df["Time"], format="%I:%M:%S %p")
    df["hour"] = df["Time"].dt.hour
    df = df.rename(
        columns={
            "Day of the week": "day_of_week",
            "Traffic Situation": "traffic_situation",
            "CarCount": "car_count",
            "BikeCount": "bike_count",
            "BusCount": "bus_count",
            "TruckCount": "truck_count",
            "Total": "traffic_volume",
        }
    )

    X = df[FEATURE_NAMES].copy()
    y = df["traffic_situation"].astype(str)

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
        "model_name": "Traffic Situation Classifier",
        "model_type": "classification",
        "feature_names": FEATURE_NAMES,
        "target_name": "traffic_situation",
        "classes": sorted(y.unique().tolist()),
        "model": pipeline,
        "metrics": metrics,
    }
    save_pickle_model(payload, MODEL_PATH)
    return payload


def load_model() -> dict:
    """Load the traffic situation model, training it if needed."""
    return ensure_model(MODEL_PATH, train_model)


def predict(input_data: dict) -> dict:
    """Predict the traffic situation using the common input payload."""
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
        "prediction_label": "Predicted traffic situation",
        "confidence_score": round(confidence_score, 4) if confidence_score is not None else None,
        "metrics": payload["metrics"],
    }

