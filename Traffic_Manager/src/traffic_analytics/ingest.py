from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from traffic_analytics.config import PipelineConfig


def _stack_expression(columns: list[str]) -> str:
    pairs = ", ".join([f"'{column}', `{column}`" for column in columns])
    return f"stack({len(columns)}, {pairs}) as (sensor_id, speed)"


def load_sensor_readings(spark: SparkSession, config: PipelineConfig) -> DataFrame:
    wide_df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(str(config.readings_path))
    )
    renamed = wide_df.withColumnRenamed(wide_df.columns[0], "timestamp")
    sensor_columns = renamed.columns[1:]
    long_df = (
        renamed.select(
            F.to_timestamp("timestamp").alias("timestamp"),
            F.expr(_stack_expression(sensor_columns)),
        )
        .withColumn("sensor_id", F.col("sensor_id").cast("string"))
        .withColumn("speed", F.col("speed").cast("double"))
        .filter(F.col("timestamp").isNotNull())
    )
    return long_df


def load_distances(spark: SparkSession, config: PipelineConfig) -> DataFrame:
    return (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(str(config.distances_path))
        .select(
            F.col("from").cast("string").alias("source_sensor"),
            F.col("to").cast("string").alias("target_sensor"),
            F.col("cost").cast("double").alias("distance_meters"),
        )
    )


def load_sensors(spark: SparkSession, config: PipelineConfig) -> DataFrame:
    path = Path(config.sensors_path)
    text_df = spark.read.text(str(path))
    return text_df.select(F.col("value").cast("string").alias("sensor_id"))
