"""Traffic ML pipeline with saved evaluation visuals and PDF output."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    avg,
    col,
    cos,
    dayofmonth,
    dayofweek,
    hour,
    lag,
    lit,
    month,
    percentile_approx,
    pi,
    sin,
    unix_timestamp,
    when,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "traffic1.csv"
OUTPUT_DIR = BASE_DIR / "output"
ML_REPORT_PDF_PATH = OUTPUT_DIR / "ml_report.pdf"


def ensure_output_dir():
    """Create the output folder used for saved ML charts and the PDF report."""
    OUTPUT_DIR.mkdir(exist_ok=True)


def build_features(df):
    """Create model features from timestamps and historical vehicle counts."""
    df = df.withColumn("DateTime", col("DateTime").cast("timestamp"))

    df = (
        df.withColumn("hour", hour("DateTime"))
        .withColumn("day_of_week", dayofweek("DateTime"))
        .withColumn("day_of_month", dayofmonth("DateTime"))
        .withColumn("month", month("DateTime"))
        .withColumn("is_weekend", when(col("day_of_week").isin(1, 7), 1).otherwise(0))
        .withColumn("hour_sin", sin((col("hour") * lit(2.0) * pi()) / lit(24.0)))
        .withColumn("hour_cos", cos((col("hour") * lit(2.0) * pi()) / lit(24.0)))
        .withColumn("timestamp_seconds", unix_timestamp("DateTime"))
    )

    ordered_window = Window.partitionBy("Junction").orderBy("DateTime")
    rolling_window = ordered_window.rowsBetween(-3, -1)

    df = (
        df.withColumn("lag_1", lag("Vehicles", 1).over(ordered_window))
        .withColumn("lag_2", lag("Vehicles", 2).over(ordered_window))
        .withColumn("lag_24", lag("Vehicles", 24).over(ordered_window))
        .withColumn("rolling_mean_3", avg("Vehicles").over(rolling_window))
    )

    return df.dropna()


def split_by_time(df):
    """Split data chronologically so the evaluation reflects future prediction."""
    split_ts = df.select(
        percentile_approx("timestamp_seconds", 0.8).alias("split_ts")
    ).first()["split_ts"]

    train = df.filter(col("timestamp_seconds") <= lit(split_ts))
    test = df.filter(col("timestamp_seconds") > lit(split_ts))

    return train, test


def evaluate_predictions(predictions):
    """Calculate common regression metrics for the saved report."""
    metrics = {}
    for metric in ("rmse", "mae", "r2"):
        evaluator = RegressionEvaluator(
            labelCol="Vehicles",
            predictionCol="prediction",
            metricName=metric,
        )
        metrics[metric] = evaluator.evaluate(predictions)
    return metrics


def save_figure(fig, filename, pdf):
    """Save a chart as both PNG and a page in the PDF report."""
    output_path = OUTPUT_DIR / filename
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    pdf.savefig(fig)
    plt.close(fig)


def save_ml_visuals(predictions, metrics, train_count, test_count):
    """Save the ML evaluation report and supporting charts."""
    pdf = predictions.select(
        "DateTime", "Junction", "Vehicles", "prediction"
    ).toPandas()
    pdf["residual"] = pdf["Vehicles"] - pdf["prediction"]
    pdf["hour_of_day"] = pdf["DateTime"].dt.hour

    hourly_actual_vs_predicted = (
        pdf.groupby("hour_of_day")[["Vehicles", "prediction"]].mean().reset_index()
    )

    with PdfPages(ML_REPORT_PDF_PATH) as report_pdf:
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        ax.set_title("Smart Traffic ML Report", fontsize=16, pad=20)
        summary_lines = [
            f"Train rows: {train_count}",
            f"Test rows: {test_count}",
            f"RMSE: {metrics['rmse']:.3f}",
            f"MAE: {metrics['mae']:.3f}",
            f"R2: {metrics['r2']:.3f}",
            "",
            "Visuals saved in the output folder:",
            "- actual_vs_predicted_ml.png",
            "- residual_distribution_ml.png",
            "- hourly_actual_vs_predicted_ml.png",
        ]
        ax.text(
            0.02,
            0.96,
            "\n".join(summary_lines),
            va="top",
            ha="left",
            fontsize=11,
            family="monospace",
        )
        report_pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(pdf["Vehicles"], pdf["prediction"], alpha=0.35, edgecolors="none")
        diagonal_min = min(pdf["Vehicles"].min(), pdf["prediction"].min())
        diagonal_max = max(pdf["Vehicles"].max(), pdf["prediction"].max())
        ax.plot(
            [diagonal_min, diagonal_max],
            [diagonal_min, diagonal_max],
            linestyle="--",
            color="red",
            linewidth=1,
        )
        ax.set_xlabel("Actual Vehicles")
        ax.set_ylabel("Predicted Vehicles")
        ax.set_title("ML Predictions: Actual vs Predicted")
        ax.grid(True, alpha=0.3)
        save_figure(fig, "actual_vs_predicted_ml.png", report_pdf)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(pdf["residual"], bins=30, color="teal", edgecolor="black", alpha=0.8)
        ax.set_xlabel("Residual (Actual - Predicted)")
        ax.set_ylabel("Frequency")
        ax.set_title("ML Residual Distribution")
        ax.grid(True, alpha=0.3)
        save_figure(fig, "residual_distribution_ml.png", report_pdf)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            hourly_actual_vs_predicted["hour_of_day"],
            hourly_actual_vs_predicted["Vehicles"],
            marker="o",
            label="Average Actual Vehicles",
        )
        ax.plot(
            hourly_actual_vs_predicted["hour_of_day"],
            hourly_actual_vs_predicted["prediction"],
            marker="o",
            label="Average Predicted Vehicles",
        )
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Average Vehicle Count")
        ax.set_title("Hourly Actual vs Predicted Traffic")
        ax.grid(True, alpha=0.3)
        ax.legend()
        save_figure(fig, "hourly_actual_vs_predicted_ml.png", report_pdf)


def main():
    """Train the model, print metrics, and save ML visuals."""
    ensure_output_dir()

    spark = (
        SparkSession.builder.appName("Smart Traffic ML")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )

    df = spark.read.csv(str(DATA_PATH), header=True, inferSchema=True)
    df = build_features(df)

    train, test = split_by_time(df)

    indexer = StringIndexer(
        inputCol="Junction",
        outputCol="JunctionIndex",
        handleInvalid="keep",
    )
    encoder = OneHotEncoder(
        inputCols=["JunctionIndex"],
        outputCols=["JunctionVec"],
        handleInvalid="keep",
    )
    assembler = VectorAssembler(
        inputCols=[
            "JunctionVec",
            "hour",
            "day_of_week",
            "day_of_month",
            "month",
            "is_weekend",
            "hour_sin",
            "hour_cos",
            "lag_1",
            "lag_2",
            "lag_24",
            "rolling_mean_3",
        ],
        outputCol="features",
    )

    model = RandomForestRegressor(
        featuresCol="features",
        labelCol="Vehicles",
        predictionCol="prediction",
        numTrees=150,
        maxDepth=12,
        minInstancesPerNode=2,
        seed=42,
    )

    pipeline = Pipeline(stages=[indexer, encoder, assembler, model])
    fitted_pipeline = pipeline.fit(train)
    predictions = fitted_pipeline.transform(test).cache()

    metrics = evaluate_predictions(predictions)
    train_count = train.count()
    test_count = test.count()

    print("=== ML Model Summary ===")
    print(f"Train rows: {train_count}")
    print(f"Test rows: {test_count}")
    print(f"RMSE: {metrics['rmse']:.3f}")
    print(f"MAE: {metrics['mae']:.3f}")
    print(f"R2: {metrics['r2']:.3f}")

    predictions = predictions.withColumn(
        "predicted_fare",
        when(col("prediction") > 50, 60)
        .when(col("prediction") > 20, 30)
        .otherwise(10),
    )

    print("\n=== Prediction Preview ===")
    predictions.select(
        "DateTime",
        "Junction",
        "Vehicles",
        "prediction",
        "predicted_fare",
    ).show(20, truncate=False)

    save_ml_visuals(predictions, metrics, train_count, test_count)

    print("\n=== Saved ML Files ===")
    print(f"ML report PDF: {ML_REPORT_PDF_PATH}")
    print(f"ML visuals folder: {OUTPUT_DIR}")

    predictions.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()
