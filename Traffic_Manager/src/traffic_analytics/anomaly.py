from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from traffic_analytics.config import PipelineConfig


def detect_anomalies(metrics_df: DataFrame, config: PipelineConfig) -> DataFrame:
    sensor_window = Window.partitionBy("sensor_id")
    anomalies = (
        metrics_df.withColumn("sensor_mean_speed", F.avg("speed").over(sensor_window))
        .withColumn(
            "sensor_std_speed",
            F.coalesce(F.stddev("speed").over(sensor_window), F.lit(0.0)),
        )
        .withColumn(
            "speed_zscore",
            F.when(
                F.col("sensor_std_speed") > 0,
                (F.col("speed") - F.col("sensor_mean_speed")) / F.col("sensor_std_speed"),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "anomaly_type",
            F.when(
                (F.abs(F.col("speed_zscore")) >= config.anomaly_zscore)
                & (F.col("speed") < F.col("avg_speed_1h")),
                F.lit("sudden_slowdown"),
            ).when(
                F.abs(F.col("speed_zscore")) >= config.anomaly_zscore,
                F.lit("sensor_outlier"),
            ),
        )
        .filter(F.col("anomaly_type").isNotNull())
    )
    return anomalies.select(
        "timestamp",
        "sensor_id",
        "speed",
        "avg_speed_1h",
        "speed_zscore",
        "anomaly_type",
        "congestion_level",
    )
