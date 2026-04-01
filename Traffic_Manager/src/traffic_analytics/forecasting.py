from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from traffic_analytics.config import PipelineConfig


def forecast_traffic(metrics_df: DataFrame, config: PipelineConfig) -> DataFrame:
    ordered = Window.partitionBy("sensor_id").orderBy("timestamp")
    trailing = ordered.rowsBetween(-config.rolling_window, 0)

    base = (
        metrics_df.withColumn("prev_speed", F.lag("speed").over(ordered))
        .withColumn(
            "speed_delta", F.coalesce(F.col("speed") - F.col("prev_speed"), F.lit(0.0))
        )
        .withColumn("trend_avg", F.avg("speed_delta").over(trailing))
        .withColumn("rolling_speed", F.avg("speed").over(trailing))
    )

    return (
        base.withColumn(
            "forecast_speed",
            F.round(
                F.greatest(
                    F.lit(0.0),
                    F.col("rolling_speed")
                    + (F.col("trend_avg") * F.lit(config.forecast_horizon)),
                ),
                2,
            ),
        )
        .withColumn(
            "forecast_density_score",
            F.round(
                F.greatest(
                    F.lit(0.0),
                    (1 - (F.col("forecast_speed") / F.col("free_flow_speed"))) * 100,
                ),
                2,
            ),
        )
        .withColumn(
            "forecast_congestion_level",
            F.when(F.col("forecast_density_score") >= 75, F.lit("severe"))
            .when(F.col("forecast_density_score") >= 55, F.lit("high"))
            .when(F.col("forecast_density_score") >= 30, F.lit("moderate"))
            .otherwise(F.lit("low")),
        )
        .select(
            "timestamp",
            "sensor_id",
            "speed",
            "forecast_speed",
            "forecast_density_score",
            "forecast_congestion_level",
        )
    )
