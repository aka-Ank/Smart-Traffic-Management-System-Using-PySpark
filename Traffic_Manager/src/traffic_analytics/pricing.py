from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def build_dynamic_pricing(metrics_df: DataFrame) -> DataFrame:
    return (
        metrics_df.withColumn(
            "demand_index",
            F.round((F.col("density_score") * 0.7) + (F.col("hour_of_day") * 0.8), 2),
        )
        .withColumn(
            "public_transport_fare_multiplier",
            F.round(
                F.when(F.col("is_peak_period"), F.lit(1.20)).otherwise(F.lit(0.90))
                + (F.col("density_proxy") * 0.35),
                2,
            ),
        )
        .withColumn(
            "congestion_charge_usd",
            F.round(
                F.when(F.col("congestion_level") == "severe", F.lit(12.0))
                .when(F.col("congestion_level") == "high", F.lit(8.0))
                .when(F.col("congestion_level") == "moderate", F.lit(4.0))
                .otherwise(F.lit(1.5)),
                2,
            ),
        )
        .withColumn(
            "parking_fee_usd",
            F.round(
                F.lit(2.0)
                + (F.col("density_proxy") * 5.0)
                + F.when(F.col("is_peak_period"), F.lit(1.5)).otherwise(F.lit(0.0)),
                2,
            ),
        )
        .select(
            "timestamp",
            "sensor_id",
            "congestion_level",
            "density_score",
            "demand_index",
            "public_transport_fare_multiplier",
            "congestion_charge_usd",
            "parking_fee_usd",
        )
    )
