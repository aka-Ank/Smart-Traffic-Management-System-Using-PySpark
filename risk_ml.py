#!/usr/bin/env python3
"""Traffic risk classification pipeline using PySpark preprocessing and scikit-learn models."""

from __future__ import annotations

import json
import os
import re
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/risk_ml_mpl_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/risk_ml_cache")


def configure_java_home() -> None:
    """Prefer Java 17 for Spark 4 when the shell default points to an older runtime."""
    current_java_home = os.environ.get("JAVA_HOME", "")
    if "17" in current_java_home or "21" in current_java_home or "25" in current_java_home:
        return

    candidates = ["17", "21", "25"]
    for version in candidates:
        try:
            result = subprocess.run(
                ["/usr/libexec/java_home", "-v", version],
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue

        java_home = result.stdout.strip()
        if java_home:
            os.environ["JAVA_HOME"] = java_home
            os.environ["PATH"] = f"{java_home}/bin:{os.environ.get('PATH', '')}"
            return


configure_java_home()

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, label_binarize

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "traffic2.csv"
OUTPUT_DIR = BASE_DIR / "output"
ARTIFACTS_DIR = OUTPUT_DIR / "risk_ml_artifacts"
MODEL_PATH = ARTIFACTS_DIR / "best_risk_model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics_summary.json"
COMPARISON_PATH = ARTIFACTS_DIR / "model_comparison.csv"
PREDICTIONS_PATH = ARTIFACTS_DIR / "sample_predictions.csv"
CONFUSION_MATRIX_PATH = ARTIFACTS_DIR / "best_model_confusion_matrix.png"
FEATURE_IMPORTANCE_PATH = ARTIFACTS_DIR / "best_model_feature_importance.png"
ROC_PATH = ARTIFACTS_DIR / "best_model_roc_curve.png"

TARGET_LABELS = ["Low", "Moderate", "High", "Severe"]
NUMERIC_COLUMNS = [
    "traffic_volume",
    "air_pollution_index",
    "humidity",
    "visibility_in_miles",
    "temperature",
    "rain_p_h",
    "snow_p_h",
    "hour",
    "weather_severity",
    "visibility_risk",
    "pollution_risk",
    "traffic_risk",
    "total_risk_score",
]
CATEGORICAL_COLUMNS = [
    "weather_type",
    "weather_description",
    "weather_category",
    "rush_hour",
]


@dataclass
class ModelResult:
    name: str
    estimator: Pipeline
    metrics: Dict[str, float]
    confusion: np.ndarray
    report: Dict[str, Dict[str, float]]
    y_pred: np.ndarray
    y_proba: np.ndarray | None


def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder.appName("TrafficRiskPrediction")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data(spark: SparkSession, csv_path: Path) -> DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    return (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(str(csv_path))
    )


def normalize_text(column: F.Column) -> F.Column:
    return F.lower(F.trim(column))


def add_weather_features(df: DataFrame) -> DataFrame:
    weather_text = normalize_text(F.coalesce(F.col("weather_description"), F.lit("unknown")))

    weather_category = (
        F.when(weather_text.rlike("thunder|storm|squall|tornado"), F.lit("Storm"))
        .when(weather_text.rlike("snow|sleet|blizzard|flurr"), F.lit("Snow"))
        .when(weather_text.rlike("fog|mist|smoke|haze|dust|sand"), F.lit("Low Visibility"))
        .when(weather_text.rlike("rain|drizzle|shower"), F.lit("Rainy"))
        .when(weather_text.rlike("cloud|overcast"), F.lit("Cloudy"))
        .when(weather_text.rlike("clear|sun"), F.lit("Clear"))
        .otherwise(F.lit("Cloudy"))
    )

    weather_severity = (
        F.when(weather_category == "Clear", F.lit(0))
        .when(weather_category == "Cloudy", F.lit(1))
        .when(weather_category == "Rainy", F.lit(2))
        .when(weather_category == "Low Visibility", F.lit(3))
        .when(weather_category == "Snow", F.lit(4))
        .when(weather_category == "Storm", F.lit(5))
        .otherwise(F.lit(1))
    )

    return df.withColumn("weather_category", weather_category).withColumn(
        "weather_severity", weather_severity
    )


def engineer_features(df: DataFrame) -> DataFrame:
    df = df.withColumn("date_time", F.to_timestamp("date_time"))

    for column in [
        "traffic_volume",
        "air_pollution_index",
        "humidity",
        "visibility_in_miles",
        "temperature",
        "rain_p_h",
        "snow_p_h",
    ]:
        df = df.withColumn(column, F.col(column).cast(DoubleType()))

    for column in ["weather_type", "weather_description"]:
        df = df.withColumn(column, F.col(column).cast(StringType()))

    df = df.withColumn("hour", F.hour("date_time"))
    df = df.withColumn(
        "rush_hour",
        F.when(
            ((F.col("hour") >= 8) & (F.col("hour") <= 10))
            | ((F.col("hour") >= 17) & (F.col("hour") <= 20)),
            F.lit("Yes"),
        ).otherwise(F.lit("No")),
    )
    df = add_weather_features(df)

    df = df.withColumn(
        "visibility_risk",
        F.when(F.col("visibility_in_miles").isNull(), F.lit(0))
        .when(F.col("visibility_in_miles") <= 1, F.lit(2))
        .when(F.col("visibility_in_miles") <= 3, F.lit(1))
        .otherwise(F.lit(0)),
    )
    df = df.withColumn(
        "pollution_risk",
        F.when(F.col("air_pollution_index").isNull(), F.lit(0))
        .when(F.col("air_pollution_index") >= 150, F.lit(2))
        .when(F.col("air_pollution_index") >= 75, F.lit(1))
        .otherwise(F.lit(0)),
    )
    df = df.withColumn(
        "traffic_risk",
        F.when(F.col("traffic_volume").isNull(), F.lit(0))
        .when(F.col("traffic_volume") >= 6000, F.lit(2))
        .when(F.col("traffic_volume") >= 3000, F.lit(1))
        .otherwise(F.lit(0)),
    )
    df = df.withColumn(
        "total_risk_score",
        F.col("visibility_risk")
        + F.col("pollution_risk")
        + F.col("traffic_risk")
        + F.when(F.col("weather_severity") >= 4, F.lit(2))
        .when(F.col("weather_severity") >= 2, F.lit(1))
        .otherwise(F.lit(0)),
    )
    df = df.withColumn(
        "risk_level",
        F.when(F.col("total_risk_score") <= 1, F.lit("Low"))
        .when((F.col("total_risk_score") >= 2) & (F.col("total_risk_score") <= 3), F.lit("Moderate"))
        .when((F.col("total_risk_score") >= 4) & (F.col("total_risk_score") <= 5), F.lit("High"))
        .otherwise(F.lit("Severe")),
    )

    return df


def handle_missing_values(df: DataFrame) -> DataFrame:
    numeric_cols = [
        "traffic_volume",
        "air_pollution_index",
        "humidity",
        "visibility_in_miles",
        "temperature",
        "rain_p_h",
        "snow_p_h",
        "hour",
    ]
    string_cols = ["weather_type", "weather_description", "weather_category", "rush_hour"]

    medians = {}
    for col_name in numeric_cols:
        median_value = df.approxQuantile(col_name, [0.5], 0.01)
        medians[col_name] = median_value[0] if median_value else 0.0

    df = df.fillna(medians)
    df = df.fillna({column: "Unknown" for column in string_cols})
    return df.dropna(subset=["date_time"])


def spark_to_pandas(df: DataFrame) -> pd.DataFrame:
    selected = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + ["risk_level"]
    pdf = df.select(*selected).toPandas()
    pdf["rush_hour"] = pdf["rush_hour"].astype(str)
    pdf["risk_level"] = pd.Categorical(pdf["risk_level"], categories=TARGET_LABELS, ordered=True)
    return pdf


def compute_class_weight_map(y: np.ndarray) -> Dict[int, float]:
    classes, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    n_classes = len(classes)
    return {cls: total / (n_classes * count) for cls, count in zip(classes, counts)}


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLUMNS),
            ("cat", categorical_transformer, CATEGORICAL_COLUMNS),
        ]
    )


def build_models(class_weight_map: Dict[int, float]) -> Dict[str, Tuple[Pipeline, Dict[str, List[object]] | None]]:
    logistic = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight=class_weight_map,
                    random_state=42,
                ),
            ),
        ]
    )

    random_forest = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_split=4,
                    class_weight=class_weight_map,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )

    if XGBOOST_AVAILABLE:
        booster = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    XGBClassifier(
                        objective="multi:softprob",
                        num_class=len(TARGET_LABELS),
                        n_estimators=250,
                        max_depth=6,
                        learning_rate=0.08,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        eval_metric="mlogloss",
                        random_state=42,
                    ),
                ),
            ]
        )
        grid = {
            "model__n_estimators": [180, 250],
            "model__max_depth": [4, 6],
            "model__learning_rate": [0.05, 0.08],
        }
        name = "XGBoost"
    else:
        booster = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", GradientBoostingClassifier(random_state=42)),
            ]
        )
        grid = {
            "model__n_estimators": [100, 150],
            "model__learning_rate": [0.05, 0.1],
        }
        name = "Gradient Boosting"

    return {
        "Logistic Regression": (logistic, None),
        "Random Forest": (random_forest, None),
        name: (booster, grid),
    }


def evaluate_model(
    name: str,
    estimator: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
) -> ModelResult:
    y_pred = estimator.predict(X_test)
    y_proba = estimator.predict_proba(X_test) if hasattr(estimator, "predict_proba") else None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }
    confusion = confusion_matrix(y_test, y_pred, labels=np.arange(len(label_encoder.classes_)))
    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    return ModelResult(
        name=name,
        estimator=estimator,
        metrics=metrics,
        confusion=confusion,
        report=report,
        y_pred=y_pred,
        y_proba=y_proba,
    )


def train_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
) -> List[ModelResult]:
    class_weight_map = compute_class_weight_map(y_train)
    models = build_models(class_weight_map)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    results: List[ModelResult] = []

    for name, (pipeline, param_grid) in models.items():
        if param_grid:
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring="f1_weighted",
                cv=cv,
                n_jobs=-1,
                verbose=0,
            )
            if name == "XGBoost" and XGBOOST_AVAILABLE:
                sample_weight = np.array([class_weight_map[label] for label in y_train])
                search.fit(X_train, y_train, model__sample_weight=sample_weight)
            else:
                search.fit(X_train, y_train)
            fitted = search.best_estimator_
        else:
            pipeline.fit(X_train, y_train)
            fitted = pipeline

        results.append(evaluate_model(name, fitted, X_test, y_test, label_encoder))

    return results


def get_feature_names(estimator: Pipeline) -> np.ndarray:
    preprocessor: ColumnTransformer = estimator.named_steps["preprocessor"]
    try:
        return preprocessor.get_feature_names_out()
    except Exception:
        names: List[str] = []
        for name, transformer, columns in preprocessor.transformers_:
            if transformer == "drop":
                continue
            if hasattr(transformer, "named_steps") and "onehot" in transformer.named_steps:
                onehot = transformer.named_steps["onehot"]
                try:
                    names.extend(onehot.get_feature_names_out(columns))
                except Exception:
                    names.extend([f"{name}_{idx}" for idx in range(len(onehot.categories_))])
            else:
                names.extend(list(columns))
        return np.array(names, dtype=object)


def plot_confusion(confusion: np.ndarray, labels: List[str], path: Path) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_feature_importance(
    result: ModelResult,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    path: Path,
) -> pd.DataFrame:
    model = result.estimator.named_steps["model"]
    transformed = result.estimator.named_steps["preprocessor"].transform(X_test.iloc[:5])
    expected_size = transformed.shape[1]
    feature_names = get_feature_names(result.estimator)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        permutation = permutation_importance(
            result.estimator,
            X_test,
            y_test,
            n_repeats=5,
            random_state=42,
            scoring="f1_weighted",
        )
        importances = permutation.importances_mean

    if len(feature_names) != len(importances):
        feature_names = np.array([f"feature_{idx}" for idx in range(expected_size)], dtype=object)
        if len(importances) != expected_size:
            aligned_size = min(expected_size, len(importances))
            feature_names = feature_names[:aligned_size]
            importances = np.asarray(importances)[:aligned_size]

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(20)
    )

    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df, x="importance", y="feature", hue="feature", dodge=False, palette="viridis", legend=False)
    plt.title(f"Top Feature Importances - {result.name}")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

    return importance_df


def plot_multiclass_roc(
    result: ModelResult,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    path: Path,
) -> float | None:
    if result.y_proba is None:
        return None

    y_bin = label_binarize(y_test, classes=np.arange(len(label_encoder.classes_)))
    roc_auc = roc_auc_score(y_bin, result.y_proba, multi_class="ovr", average="weighted")

    plt.figure(figsize=(8, 6))
    plotted = False
    for idx, class_name in enumerate(label_encoder.classes_):
        if len(np.unique(y_bin[:, idx])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, idx], result.y_proba[:, idx])
        plt.plot(fpr, tpr, label=class_name)
        plotted = True

    if plotted:
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.title(f"ROC Curve - {result.name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=200)
    plt.close()
    return float(roc_auc)


def save_model(result: ModelResult, label_encoder: LabelEncoder) -> None:
    joblib.dump(
        {
            "model_name": result.name,
            "pipeline": result.estimator,
            "label_encoder": label_encoder,
            "target_labels": TARGET_LABELS,
            "numeric_columns": NUMERIC_COLUMNS,
            "categorical_columns": CATEGORICAL_COLUMNS,
        },
        MODEL_PATH,
    )


def save_metrics(
    results: List[ModelResult],
    best_result: ModelResult,
    best_roc_auc: float | None,
    importance_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    payload = {"best_model": best_result.name, "models": {}}

    for result in results:
        rows.append({"model": result.name, **result.metrics})
        payload["models"][result.name] = {
            **result.metrics,
            "classification_report": result.report,
            "confusion_matrix": result.confusion.tolist(),
        }

    if best_roc_auc is not None:
        payload["models"][best_result.name]["roc_auc_ovr_weighted"] = best_roc_auc
    payload["feature_importance_top20"] = importance_df.to_dict(orient="records")

    comparison_df = pd.DataFrame(rows).sort_values(["f1_score", "accuracy"], ascending=False)
    comparison_df.to_csv(COMPARISON_PATH, index=False)
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return comparison_df


def print_results(
    comparison_df: pd.DataFrame,
    best_result: ModelResult,
    best_roc_auc: float | None,
    sample_predictions: pd.DataFrame,
    importance_df: pd.DataFrame,
    label_encoder: LabelEncoder,
) -> None:
    print("\nModel Comparison")
    print(comparison_df.to_string(index=False))

    print(f"\nBest Model: {best_result.name}")
    if best_roc_auc is not None:
        print(f"Weighted ROC-AUC (OvR): {best_roc_auc:.4f}")

    print("\nClassification Report")
    print(pd.DataFrame(best_result.report).transpose().round(4).to_string())

    print("\nConfusion Matrix")
    confusion_df = pd.DataFrame(
        best_result.confusion,
        index=[f"Actual_{label}" for label in label_encoder.classes_],
        columns=[f"Pred_{label}" for label in label_encoder.classes_],
    )
    print(confusion_df.to_string())

    print("\nTop Feature Importances")
    print(importance_df.head(10).round(6).to_string(index=False))

    print("\nSample Predictions")
    print(sample_predictions.to_string(index=False))

    print(f"\nSaved best model to: {MODEL_PATH}")
    print(f"Saved artifacts to: {ARTIFACTS_DIR}")


def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [re.sub(r"[^0-9a-zA-Z_]+", "_", column).strip("_") for column in df.columns]
    return df


def main() -> None:
    ensure_output_dirs()
    spark = create_spark_session()

    try:
        spark_df = load_data(spark, DATA_PATH)
        spark_df = engineer_features(spark_df)
        spark_df = handle_missing_values(spark_df)
        pandas_df = sanitize_columns(spark_to_pandas(spark_df))

        X = pandas_df[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS].copy()
        y = pandas_df["risk_level"].astype(str).copy()

        label_encoder = LabelEncoder()
        label_encoder.fit(TARGET_LABELS)
        y_encoded = label_encoder.transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_encoded,
            test_size=0.2,
            random_state=42,
            stratify=y_encoded,
        )

        results = train_models(X_train, X_test, y_train, y_test, label_encoder)
        best_result = max(results, key=lambda item: (item.metrics["f1_score"], item.metrics["accuracy"]))

        plot_confusion(best_result.confusion, list(label_encoder.classes_), CONFUSION_MATRIX_PATH)
        importance_df = plot_feature_importance(best_result, X_test, y_test, FEATURE_IMPORTANCE_PATH)
        best_roc_auc = plot_multiclass_roc(best_result, y_test, label_encoder, ROC_PATH)
        save_model(best_result, label_encoder)

        sample_predictions = X_test.copy()
        sample_predictions["actual_risk_level"] = label_encoder.inverse_transform(y_test)
        sample_predictions["predicted_risk_level"] = label_encoder.inverse_transform(best_result.y_pred)
        sample_predictions = sample_predictions.head(15)
        sample_predictions.to_csv(PREDICTIONS_PATH, index=False)

        comparison_df = save_metrics(results, best_result, best_roc_auc, importance_df)
        print_results(
            comparison_df,
            best_result,
            best_roc_auc,
            sample_predictions[["actual_risk_level", "predicted_risk_level"]],
            importance_df,
            label_encoder,
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
