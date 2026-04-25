"""Spark job 2: traffic risk analysis from environmental data."""

from __future__ import annotations

from typing import Any

from pyspark.sql import Window, functions as F

from spark_jobs.utils import DATA_DIR, create_spark_session


DATA_PATH = DATA_DIR / "traffic2.csv"


def run_job() -> dict[str, Any]:
    """Run the second Spark job and return JSON-safe risk information."""
    spark = create_spark_session("SparkJob2RiskAnalysis")
    try:
        df = spark.read.csv(str(DATA_PATH), header=True, inferSchema=True)
        df = df.withColumn("date_time", F.to_timestamp("date_time"))
        df = df.withColumn("hour", F.hour("date_time"))
        weather_desc = F.lower(F.coalesce(F.col("weather_description"), F.lit("unknown")))

        df = df.withColumn(
            "weather_category",
            F.when(weather_desc.contains("clear"), "Clear")
            .when(weather_desc.contains("cloud"), "Cloudy")
            .when(weather_desc.rlike("rain|drizzle"), "Rainy")
            .when(weather_desc.rlike("fog|mist|haze|smoke"), "Low Visibility")
            .when(weather_desc.rlike("thunder|storm"), "Storm")
            .otherwise("Other"),
        )
        df = df.withColumn(
            "weather_severity",
            F.when(F.col("weather_category") == "Clear", 0)
            .when(F.col("weather_category") == "Cloudy", 1)
            .when(F.col("weather_category") == "Rainy", 2)
            .when(F.col("weather_category") == "Low Visibility", 3)
            .when(F.col("weather_category") == "Storm", 4)
            .otherwise(1),
        )
        df = df.withColumn(
            "visibility_risk",
            F.when(F.col("visibility_in_miles") < 2, 2)
            .when(F.col("visibility_in_miles") < 5, 1)
            .otherwise(0),
        )
        df = df.withColumn(
            "pollution_risk",
            F.when(F.col("air_pollution_index") > 150, 2)
            .when(F.col("air_pollution_index") > 100, 1)
            .otherwise(0),
        )
        df = df.withColumn(
            "traffic_risk",
            F.when(F.col("traffic_volume") > 6000, 2)
            .when(F.col("traffic_volume") > 3000, 1)
            .otherwise(0),
        )
        df = df.withColumn(
            "total_risk_score",
            F.col("weather_severity")
            + F.col("visibility_risk")
            + F.col("pollution_risk")
            + F.col("traffic_risk"),
        )
        df = df.withColumn(
            "risk_level",
            F.when(F.col("total_risk_score") >= 6, "Severe")
            .when(F.col("total_risk_score") >= 4, "High")
            .when(F.col("total_risk_score") >= 2, "Moderate")
            .otherwise("Low"),
        )

        most_risky_hour = (
            df.groupBy("hour")
            .agg(F.avg("total_risk_score").alias("avg_risk_score"))
            .orderBy(F.col("avg_risk_score").desc())
            .first()
        )
        safest_hour = (
            df.groupBy("hour")
            .agg(F.avg("total_risk_score").alias("avg_risk_score"))
            .orderBy("avg_risk_score")
            .first()
        )

        risk_rows = (
            df.groupBy("risk_level")
            .agg(
                F.count("*").alias("records"),
                F.avg("traffic_volume").alias("avg_traffic_volume"),
                F.avg("visibility_in_miles").alias("avg_visibility"),
            )
            .orderBy("risk_level")
            .collect()
        )

        transition_window = Window.orderBy("date_time")
        transitions = (
            df.withColumn("previous_risk_level", F.lag("risk_level").over(transition_window))
            .filter(F.col("risk_level") != F.col("previous_risk_level"))
            .select("date_time", "previous_risk_level", "risk_level", "traffic_volume")
            .limit(10)
            .collect()
        )

        return {
            "job_name": "Spark Job 2",
            "description": "Traffic risk scoring using weather, visibility, pollution, and traffic volume.",
            "summary": {
                "most_risky_hour": int(most_risky_hour["hour"]),
                "highest_average_risk_score": round(float(most_risky_hour["avg_risk_score"]), 2),
                "safest_hour": int(safest_hour["hour"]),
                "lowest_average_risk_score": round(float(safest_hour["avg_risk_score"]), 2),
            },
            "rows": [
                {
                    "risk_level": row["risk_level"],
                    "records": int(row["records"]),
                    "avg_traffic_volume": round(float(row["avg_traffic_volume"]), 2),
                    "avg_visibility": round(float(row["avg_visibility"]), 2),
                }
                for row in risk_rows
            ],
            "transitions": [
                {
                    "date_time": str(row["date_time"]),
                    "previous_risk_level": row["previous_risk_level"] or "Start",
                    "risk_level": row["risk_level"],
                    "traffic_volume": float(row["traffic_volume"]),
                }
                for row in transitions
            ],
        }
    finally:
        spark.stop()

