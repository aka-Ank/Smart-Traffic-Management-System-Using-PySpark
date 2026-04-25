from __future__ import annotations

import json
from pathlib import Path

from traffic_analytics.anomaly import detect_anomalies
from traffic_analytics.config import PipelineConfig
from traffic_analytics.forecasting import forecast_traffic
from traffic_analytics.ingest import load_distances, load_sensor_readings, load_sensors
from traffic_analytics.metrics import build_traffic_metrics, detect_congestion
from traffic_analytics.pricing import build_dynamic_pricing
from traffic_analytics.routing import build_route_recommendation
from traffic_analytics.session import build_spark


def run_pipeline(config: PipelineConfig) -> dict:
    spark = build_spark(f"TrafficAnalytics-{config.dataset}")
    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        readings_df = load_sensor_readings(spark, config)
        distances_df = load_distances(spark, config)
        sensors_df = load_sensors(spark, config)

        metrics_df = build_traffic_metrics(readings_df.join(sensors_df, "sensor_id"), config).cache()
        congestion_df = detect_congestion(metrics_df)
        forecast_df = forecast_traffic(metrics_df, config)
        anomalies_df = detect_anomalies(metrics_df, config)
        pricing_df = build_dynamic_pricing(metrics_df)

        metrics_df.write.mode("overwrite").parquet(str(output_root / "traffic_metrics.parquet"))
        congestion_df.write.mode("overwrite").parquet(str(output_root / "congestion_alerts.parquet"))
        forecast_df.write.mode("overwrite").parquet(str(output_root / "traffic_forecast.parquet"))
        anomalies_df.write.mode("overwrite").parquet(str(output_root / "anomalies.parquet"))
        pricing_df.write.mode("overwrite").parquet(str(output_root / "pricing_recommendations.parquet"))

        top_sensors = (
            congestion_df.groupBy("sensor_id")
            .count()
            .orderBy("count", ascending=False)
            .limit(10)
            .collect()
        )

        summary = {
            "dataset": config.dataset,
            "records_processed": metrics_df.count(),
            "congestion_alerts": congestion_df.count(),
            "forecast_records": forecast_df.count(),
            "anomalies_detected": anomalies_df.count(),
            "pricing_records": pricing_df.count(),
            "top_congested_sensors": [
                {"sensor_id": row["sensor_id"], "events": row["count"]} for row in top_sensors
            ],
        }

        (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    finally:
        spark.stop()


def optimize_route(config: PipelineConfig, source: str, target: str) -> dict:
    spark = build_spark(f"RouteOptimization-{config.dataset}")
    try:
        readings_df = load_sensor_readings(spark, config)
        sensors_df = load_sensors(spark, config)
        metrics_df = build_traffic_metrics(readings_df.join(sensors_df, "sensor_id"), config)
        distances_df = load_distances(spark, config)
        return build_route_recommendation(distances_df, metrics_df, source, target)
    finally:
        spark.stop()
