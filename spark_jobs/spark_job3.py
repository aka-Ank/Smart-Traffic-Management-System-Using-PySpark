"""Spark job 3: vehicle composition and traffic situation summary."""

from __future__ import annotations

from typing import Any

from pyspark.sql import functions as F

from spark_jobs.utils import DATA_DIR, create_spark_session


DATA_PATH = DATA_DIR / "traffic3.csv"


def run_job() -> dict[str, Any]:
    """Run the third Spark job and return JSON-safe vehicle statistics."""
    spark = create_spark_session("SparkJob3VehicleAnalysis")
    try:
        df = spark.read.csv(str(DATA_PATH), header=True, inferSchema=True)
        df = df.withColumnRenamed("Day of the week", "DayOfWeek")
        df = df.withColumnRenamed("Traffic Situation", "TrafficSituation")

        totals = df.agg(
            F.sum("CarCount").alias("total_cars"),
            F.sum("BikeCount").alias("total_bikes"),
            F.sum("BusCount").alias("total_buses"),
            F.sum("TruckCount").alias("total_trucks"),
            F.sum("Total").alias("grand_total"),
            F.count("*").alias("records"),
        ).first()

        traffic_rows = (
            df.groupBy("TrafficSituation")
            .agg(
                F.count("*").alias("records"),
                F.avg("Total").alias("average_total_vehicles"),
            )
            .orderBy("TrafficSituation")
            .collect()
        )

        weekday_rows = (
            df.groupBy("DayOfWeek")
            .agg(F.avg("Total").alias("average_total_vehicles"))
            .orderBy(F.col("average_total_vehicles").desc())
            .collect()
        )

        return {
            "job_name": "Spark Job 3",
            "description": "Vehicle composition and traffic situation summary.",
            "summary": {
                "records": int(totals["records"]),
                "total_cars": float(totals["total_cars"]),
                "total_bikes": float(totals["total_bikes"]),
                "total_buses": float(totals["total_buses"]),
                "total_trucks": float(totals["total_trucks"]),
                "grand_total": float(totals["grand_total"]),
            },
            "rows": [
                {
                    "traffic_situation": row["TrafficSituation"],
                    "records": int(row["records"]),
                    "average_total_vehicles": round(float(row["average_total_vehicles"]), 2),
                }
                for row in traffic_rows
            ],
            "weekday_summary": [
                {
                    "day_of_week": row["DayOfWeek"],
                    "average_total_vehicles": round(float(row["average_total_vehicles"]), 2),
                }
                for row in weekday_rows
            ],
        }
    finally:
        spark.stop()

