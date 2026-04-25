from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from traffic_analytics.config import PipelineConfig


def build_traffic_metrics(readings_df: DataFrame, config: PipelineConfig) -> DataFrame:
    sensor_window = Window.partitionBy("sensor_id")
    ordered_window = (
        Window.partitionBy("sensor_id")
        .orderBy("timestamp")
        .rowsBetween(-config.rolling_window, 0)
    )

    metrics_df = (
        readings_df.withColumn(
            "free_flow_speed",
            F.percentile_approx("speed", 0.95).over(sensor_window),
        )
        .withColumn("avg_speed_1h", F.avg("speed").over(ordered_window))
        .withColumn(
            "speed_std_1h",
            F.coalesce(F.stddev("speed").over(ordered_window), F.lit(0.0)),
        )
        .withColumn(
            "density_proxy",
            F.when(
                F.col("free_flow_speed") > 0,
                F.greatest(
                    F.lit(0.0),
                    1 - (F.col("speed") / F.col("free_flow_speed")),
                ),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn("density_score", F.round(F.col("density_proxy") * 100, 2))
        .withColumn(
            "congestion_level",
            F.when(F.col("density_proxy") >= 0.75, F.lit("severe"))
            .when(F.col("density_proxy") >= config.congestion_threshold, F.lit("high"))
            .when(F.col("density_proxy") >= 0.30, F.lit("moderate"))
            .otherwise(F.lit("low")),
        )
        .withColumn(
            "travel_time_multiplier",
            F.round(1 + (F.col("density_proxy") * 2.5), 3),
        )
        .withColumn("hour_of_day", F.hour("timestamp"))
        .withColumn(
            "is_peak_period",
            (
                F.col("hour_of_day").between(
                    config.peak_start_hour, config.peak_end_hour
                )
                | F.col("hour_of_day").between(
                    config.evening_peak_start_hour, config.evening_peak_end_hour
                )
            ).cast("boolean"),
        )
    )
    return metrics_df


def detect_congestion(metrics_df: DataFrame) -> DataFrame:
    return metrics_df.filter(
        (F.col("congestion_level").isin("high", "severe"))
        | (F.col("travel_time_multiplier") >= 2.0)
    ).select(
        "timestamp",
        "sensor_id",
        "speed",
        "free_flow_speed",
        "density_score",
        "congestion_level",
        "travel_time_multiplier",
        "is_peak_period",
    )
