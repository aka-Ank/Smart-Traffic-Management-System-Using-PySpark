"""Spark job 1: traffic summary and dynamic pricing insights."""

from __future__ import annotations

from typing import Any

from pyspark.sql import functions as F

from spark_jobs.utils import DATA_DIR, create_spark_session


DATA_PATH = DATA_DIR / "traffic1.csv"


def run_job() -> dict[str, Any]:
    """Run the first Spark job and return JSON-safe summary data."""
    spark = create_spark_session("SparkJob1TrafficSummary")
    try:
        df = spark.read.csv(str(DATA_PATH), header=True, inferSchema=True)
        df = df.withColumn("DateTime", F.col("DateTime").cast("timestamp"))
        df = df.withColumn("hour_of_day", F.hour("DateTime"))
        df = df.withColumn(
            "traffic_level",
            F.when(F.col("Vehicles") <= 20, "Low Traffic")
            .when(F.col("Vehicles") <= 50, "Medium Traffic")
            .otherwise("High Traffic"),
        )
        df = df.withColumn(
            "dynamic_fare",
            F.when(F.col("Vehicles") > 50, 60)
            .when(F.col("Vehicles") > 20, 30)
            .otherwise(10),
        )

        summary_row = df.agg(
            F.count("*").alias("records"),
            F.sum("Vehicles").alias("total_vehicles"),
            F.avg("Vehicles").alias("average_vehicles"),
            F.sum("dynamic_fare").alias("total_dynamic_fare"),
        ).first()

        best_hour = (
            df.groupBy("hour_of_day")
            .agg(F.avg("Vehicles").alias("avg_vehicles"))
            .orderBy("avg_vehicles")
            .first()
        )
        worst_hour = (
            df.groupBy("hour_of_day")
            .agg(F.avg("Vehicles").alias("avg_vehicles"))
            .orderBy(F.col("avg_vehicles").desc())
            .first()
        )

        hourly_rows = (
            df.groupBy("hour_of_day")
            .agg(
                F.sum("Vehicles").alias("total_vehicles"),
                F.avg("Vehicles").alias("average_vehicles"),
                F.sum("dynamic_fare").alias("total_dynamic_fare"),
            )
            .orderBy("hour_of_day")
            .limit(12)
            .collect()
        )

        return {
            "job_name": "Spark Job 1",
            "description": "Traffic volume summary with dynamic pricing insights.",
            "summary": {
                "records": int(summary_row["records"]),
                "total_vehicles": float(summary_row["total_vehicles"]),
                "average_vehicles": round(float(summary_row["average_vehicles"]), 2),
                "total_dynamic_fare": float(summary_row["total_dynamic_fare"]),
                "best_hour_to_travel": int(best_hour["hour_of_day"]),
                "worst_hour_to_travel": int(worst_hour["hour_of_day"]),
            },
            "rows": [
                {
                    "hour_of_day": int(row["hour_of_day"]),
                    "total_vehicles": float(row["total_vehicles"]),
                    "average_vehicles": round(float(row["average_vehicles"]), 2),
                    "total_dynamic_fare": float(row["total_dynamic_fare"]),
                }
                for row in hourly_rows
            ],
        }
    finally:
        spark.stop()

