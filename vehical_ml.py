"""PySpark classification pipeline for traffic situation analysis."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, OneVsRest, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession, functions as F


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "output" / "traffic3.csv"
MODEL_OUTPUT_DIR = BASE_DIR / "output" / "best_traffic_situation_model"
OUTPUT_DIR = BASE_DIR / "output"
RISK_PDF_PATH = OUTPUT_DIR / "traffic.pdf"
MODEL_COMPARISON_PATH = OUTPUT_DIR / "risk_model_comparison.png"
RF_CONFUSION_PATH = OUTPUT_DIR / "risk_rf_confusion_matrix.png"
RF_IMPORTANCE_PATH = OUTPUT_DIR / "risk_rf_feature_importance.png"
REQUIRED_COLUMNS = {
    "Time",
    "Date",
    "DayOfWeek",
    "CarCount",
    "BikeCount",
    "BusCount",
    "TruckCount",
    "Total",
    "TrafficSituation",
}
CANONICAL_NAME_MAP = {
    "time": "Time",
    "date": "Date",
    "dayofweek": "DayOfWeek",
    "dayoftheweek": "DayOfWeek",
    "carcount": "CarCount",
    "bikecount": "BikeCount",
    "buscount": "BusCount",
    "truckcount": "TruckCount",
    "total": "Total",
    "trafficsituation": "TrafficSituation",
}


def resolve_data_path():
    """Return the required dataset path after validating its schema."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Expected dataset not found at: {DATA_PATH}. "
            "Please place traffic3.csv in the output folder."
        )

    with DATA_PATH.open(newline="") as csv_file:
        header = next(csv.reader(csv_file), [])

    sanitized_header = {sanitize_column_name(name) for name in header}
    if not REQUIRED_COLUMNS.issubset(sanitized_header):
        raise ValueError(
            "The CSV at output/traffic3.csv does not contain the required columns: "
            + ", ".join(sorted(REQUIRED_COLUMNS))
        )

    return DATA_PATH


def sanitize_column_name(name):
    """Convert raw CSV headers into Spark-friendly names."""
    canonical_key = "".join(ch.lower() for ch in name if ch.isalnum())
    if canonical_key in CANONICAL_NAME_MAP:
        return CANONICAL_NAME_MAP[canonical_key]

    cleaned = name.strip().replace(" ", "_").replace("/", "_")
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in cleaned)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def parse_time_to_hour(df):
    """Convert the Time column into an integer hour feature."""
    time_as_text = F.col("Time").cast("string")
    hour_part = F.regexp_extract(time_as_text, r"^\s*(\d{1,2})", 1).cast("int")
    meridiem = F.upper(F.regexp_extract(time_as_text, r"(?i)\b(AM|PM)\b", 1))

    parsed_hour = (
        F.when(hour_part.isNull(), F.lit(None))
        .when((meridiem == "AM") & (hour_part == 12), F.lit(0))
        .when((meridiem == "PM") & (hour_part < 12), hour_part + 12)
        .otherwise(hour_part)
    )
    return df.withColumn("Time_hour", parsed_hour.cast("int"))


def prepare_dataframe(df):
    """Clean column names, handle nulls, and build requested features."""
    renamed_df = df
    for old_name in df.columns:
        new_name = sanitize_column_name(old_name)
        if old_name != new_name:
            renamed_df = renamed_df.withColumnRenamed(old_name, new_name)

    required_columns = sorted(REQUIRED_COLUMNS)
    missing_columns = [col_name for col_name in required_columns if col_name not in renamed_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    numeric_columns = ["CarCount", "BikeCount", "BusCount", "TruckCount", "Total"]
    for column_name in numeric_columns:
        renamed_df = renamed_df.withColumn(column_name, F.col(column_name).cast("double"))

    # Remove rows that cannot be used for modeling, then fill remaining numeric nulls safely.
    cleaned_df = (
        renamed_df.dropna(subset=["Time", "DayOfWeek", "TrafficSituation"])
        .fillna(0, subset=numeric_columns)
        .transform(parse_time_to_hour)
        .dropna(subset=["Time_hour"])
    )

    total_safe = F.when(F.col("Total") > 0, F.col("Total")).otherwise(F.lit(None))
    engineered_df = (
        cleaned_df.withColumn("Car_ratio", F.coalesce(F.col("CarCount") / total_safe, F.lit(0.0)))
        .withColumn(
            "Heavy_ratio",
            F.coalesce((F.col("BusCount") + F.col("TruckCount")) / total_safe, F.lit(0.0)),
        )
    )

    return engineered_df


def build_pipeline(classifier):
    """Create a reusable ML pipeline with preprocessing plus classifier."""
    day_indexer = StringIndexer(
        inputCol="DayOfWeek",
        outputCol="DayOfWeek_index",
        handleInvalid="keep",
    )
    day_encoder = OneHotEncoder(
        inputCols=["DayOfWeek_index"],
        outputCols=["DayOfWeek_encoded"],
        handleInvalid="keep",
    )
    label_indexer = StringIndexer(
        inputCol="TrafficSituation",
        outputCol="label",
        handleInvalid="keep",
    )
    assembler = VectorAssembler(
        inputCols=[
            "CarCount",
            "BikeCount",
            "BusCount",
            "TruckCount",
            "Total",
            "Car_ratio",
            "Heavy_ratio",
            "Time_hour",
            "DayOfWeek_encoded",
        ],
        outputCol="features",
        handleInvalid="keep",
    )

    return Pipeline(stages=[day_indexer, day_encoder, label_indexer, assembler, classifier])


def evaluate_model(name, pipeline_model, test_df):
    """Evaluate a fitted model and print requested outputs."""
    predictions = pipeline_model.transform(test_df).cache()

    accuracy_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy",
    )
    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1",
    )

    accuracy = accuracy_evaluator.evaluate(predictions)
    f1_score = f1_evaluator.evaluate(predictions)

    print(f"\n{'=' * 80}")
    print(f"{name} Evaluation")
    print(f"{'=' * 80}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

    label_model = pipeline_model.stages[2]
    labels = list(label_model.labels)
    mapping_expr = F.create_map(
        *[item for label, idx in [(label, i) for i, label in enumerate(labels)] for item in (F.lit(float(idx)), F.lit(label))]
    )

    readable_predictions = (
        predictions.withColumn("actual_label", mapping_expr[F.col("label")])
        .withColumn("predicted_label", mapping_expr[F.col("prediction")])
    )

    print("\nSample predictions (actual vs predicted):")
    readable_predictions.select(
        "Time",
        "Date",
        "DayOfWeek",
        "actual_label",
        "predicted_label",
    ).show(10, truncate=False)

    print("Confusion matrix:")
    confusion_df = (
        readable_predictions.groupBy("actual_label")
        .pivot("predicted_label", labels)
        .count()
        .fillna(0)
        .orderBy("actual_label")
    )
    confusion_df.show(truncate=False)

    confusion_rows = confusion_df.collect()
    confusion_matrix = []
    predicted_labels = labels
    for row in confusion_rows:
        confusion_matrix.append([int(row[predicted]) if row[predicted] is not None else 0 for predicted in predicted_labels])

    return {
        "name": name,
        "model": pipeline_model,
        "predictions": predictions,
        "accuracy": accuracy,
        "f1_score": f1_score,
        "labels": labels,
        "confusion_matrix": confusion_matrix,
    }


def show_random_forest_feature_importance(rf_result):
    """Extract and print Random Forest feature importances."""
    rf_model = rf_result["model"].stages[-1]
    day_indexer_model = rf_result["model"].stages[0]
    day_labels = list(day_indexer_model.labels)
    encoded_labels = day_labels[:-1] if len(day_labels) > 1 else day_labels

    feature_names = [
        "CarCount",
        "BikeCount",
        "BusCount",
        "TruckCount",
        "Total",
        "Car_ratio",
        "Heavy_ratio",
        "Time_hour",
    ] + [f"DayOfWeek_encoded_{label}" for label in encoded_labels]

    importances = list(rf_model.featureImportances.toArray())

    print("\nRandom Forest feature importance:")
    for feature_name, importance in sorted(
        zip(feature_names, importances),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"{feature_name:<28} {importance:.6f}")

    return sorted(
        zip(feature_names, importances),
        key=lambda item: item[1],
        reverse=True,
    )


def build_report(summary_lines, rf_result, gbt_result, best_result, rf_importances):
    """Save graphs and a PDF summary report."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    with PdfPages(RISK_PDF_PATH) as pdf:
        # Summary page
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(
            0.02,
            0.98,
            "\n".join(summary_lines),
            va="top",
            ha="left",
            fontsize=12,
            family="monospace",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Model comparison graph
        model_names = [rf_result["name"], gbt_result["name"]]
        accuracies = [rf_result["accuracy"], gbt_result["accuracy"]]
        f1_scores = [rf_result["f1_score"], gbt_result["f1_score"]]

        fig, ax = plt.subplots(figsize=(10, 6))
        positions = range(len(model_names))
        bar_width = 0.35
        ax.bar([x - bar_width / 2 for x in positions], accuracies, width=bar_width, label="Accuracy")
        ax.bar([x + bar_width / 2 for x in positions], f1_scores, width=bar_width, label="F1 Score")
        ax.set_xticks(list(positions))
        ax.set_xticklabels(model_names, rotation=10, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_title("Traffic Situation Model Comparison")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(MODEL_COMPARISON_PATH, dpi=200, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Random Forest confusion matrix graph
        fig, ax = plt.subplots(figsize=(8, 6))
        matrix = rf_result["confusion_matrix"]
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_title("Random Forest Confusion Matrix")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")
        ax.set_xticks(range(len(rf_result["labels"])))
        ax.set_yticks(range(len(rf_result["labels"])))
        ax.set_xticklabels(rf_result["labels"], rotation=45, ha="right")
        ax.set_yticklabels(rf_result["labels"])
        for row_index, row_values in enumerate(matrix):
            for col_index, value in enumerate(row_values):
                ax.text(col_index, row_index, str(value), ha="center", va="center", color="black")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(RF_CONFUSION_PATH, dpi=200, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Random Forest feature importance graph
        top_importances = rf_importances[:10]
        feature_labels = [item[0] for item in reversed(top_importances)]
        feature_scores = [item[1] for item in reversed(top_importances)]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(feature_labels, feature_scores, color="#2a6f97")
        ax.set_xlabel("Importance")
        ax.set_title("Random Forest Feature Importance")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(RF_IMPORTANCE_PATH, dpi=200, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved report PDF to: {RISK_PDF_PATH}")
    print(f"Saved graph to: {MODEL_COMPARISON_PATH}")
    print(f"Saved graph to: {RF_CONFUSION_PATH}")
    print(f"Saved graph to: {RF_IMPORTANCE_PATH}")


def save_best_model(best_result):
    """Persist the winning pipeline model to disk."""
    if MODEL_OUTPUT_DIR.exists():
        # Overwrite any earlier saved best model.
        import shutil

        shutil.rmtree(MODEL_OUTPUT_DIR)

    best_result["model"].write().overwrite().save(str(MODEL_OUTPUT_DIR))
    print(f"\nBest model saved to: {MODEL_OUTPUT_DIR}")


def main():
    """Run the traffic situation classification workflow end-to-end."""
    data_path = resolve_data_path()
    print(f"Reading dataset from: {data_path}")

    spark = (
        SparkSession.builder.appName("TrafficSituationClassification")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        raw_df = spark.read.csv(str(data_path), header=True, inferSchema=True)

        print("\nRaw schema:")
        raw_df.printSchema()

        prepared_df = prepare_dataframe(raw_df).cache()

        print("\nPrepared data preview:")
        prepared_df.select(
            "Time",
            "Date",
            "DayOfWeek",
            "CarCount",
            "BikeCount",
            "BusCount",
            "TruckCount",
            "Total",
            "Car_ratio",
            "Heavy_ratio",
            "Time_hour",
            "TrafficSituation",
        ).show(10, truncate=False)

        print(f"Prepared row count: {prepared_df.count()}")

        distinct_label_count = prepared_df.select("TrafficSituation").distinct().count()
        print(f"Distinct TrafficSituation classes: {distinct_label_count}")

        train_df, test_df = prepared_df.randomSplit([0.8, 0.2], seed=42)
        train_df = train_df.cache()
        test_df = test_df.cache()

        print(f"Training rows: {train_df.count()}")
        print(f"Testing rows: {test_df.count()}")

        rf_pipeline = build_pipeline(
            RandomForestClassifier(
                labelCol="label",
                featuresCol="features",
                predictionCol="prediction",
                probabilityCol="probability",
                rawPredictionCol="rawPrediction",
                numTrees=120,
                maxDepth=10,
                seed=42,
            )
        )
        gbt_classifier = GBTClassifier(
            labelCol="label",
            featuresCol="features",
            predictionCol="prediction",
            maxIter=80,
            maxDepth=6,
            seed=42,
        )
        if distinct_label_count > 2:
            print(
                "TrafficSituation has more than two classes, "
                "so GBTClassifier is used inside OneVsRest for multiclass support."
            )
            gbt_estimator = OneVsRest(
                labelCol="label",
                featuresCol="features",
                predictionCol="prediction",
                classifier=gbt_classifier,
            )
            gbt_name = "GBTClassifier (OneVsRest)"
        else:
            gbt_estimator = gbt_classifier
            gbt_name = "GBTClassifier"

        gbt_pipeline = build_pipeline(gbt_estimator)

        OUTPUT_DIR.mkdir(exist_ok=True)

        rf_model = rf_pipeline.fit(train_df)
        rf_result = evaluate_model("RandomForestClassifier", rf_model, test_df)

        gbt_model = gbt_pipeline.fit(train_df)
        gbt_result = evaluate_model(gbt_name, gbt_model, test_df)

        print(f"\n{'=' * 80}")
        print("Model comparison")
        print(f"{'=' * 80}")
        print(f"RandomForestClassifier Accuracy: {rf_result['accuracy']:.4f}")
        print(f"GBTClassifier Accuracy:          {gbt_result['accuracy']:.4f}")

        best_result = max([rf_result, gbt_result], key=lambda item: (item["accuracy"], item["f1_score"]))
        print(
            f"Best model: {best_result['name']} "
            f"(Accuracy={best_result['accuracy']:.4f}, F1={best_result['f1_score']:.4f})"
        )

        rf_importances = show_random_forest_feature_importance(rf_result)

        save_best_model(best_result)

        summary_lines = [
            "Traffic Situation Classification Report",
            "",
            f"Dataset: {data_path}",
            f"Prepared rows: {prepared_df.count()}",
            f"Training rows: {train_df.count()}",
            f"Testing rows: {test_df.count()}",
            f"Distinct classes: {distinct_label_count}",
            "",
            f"Random Forest Accuracy: {rf_result['accuracy']:.4f}",
            f"Random Forest F1 Score: {rf_result['f1_score']:.4f}",
            f"GBT Accuracy: {gbt_result['accuracy']:.4f}",
            f"GBT F1 Score: {gbt_result['f1_score']:.4f}",
            "",
            f"Best Model: {best_result['name']}",
            f"Best Model Accuracy: {best_result['accuracy']:.4f}",
            f"Best Model F1 Score: {best_result['f1_score']:.4f}",
            f"Saved Best Model: {MODEL_OUTPUT_DIR}",
        ]
        build_report(summary_lines, rf_result, gbt_result, best_result, rf_importances)

        print("\nFeature vector preview from best model:")
        (
            best_result["predictions"]
            .withColumn("features_array", vector_to_array("features"))
            .select("features_array", "prediction", "label")
            .show(5, truncate=False)
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
