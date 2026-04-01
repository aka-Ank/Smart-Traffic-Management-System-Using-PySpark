from __future__ import annotations

from collections import defaultdict
import heapq

from pyspark.sql import DataFrame, functions as F


def build_route_recommendation(
    distances_df: DataFrame, metrics_df: DataFrame, source: str, target: str
) -> dict:
    latest_timestamp = metrics_df.agg(F.max("timestamp").alias("max_ts")).collect()[0]["max_ts"]
    latest_metrics = (
        metrics_df.filter(F.col("timestamp") == F.lit(latest_timestamp))
        .select("sensor_id", "travel_time_multiplier", "congestion_level")
        .collect()
    )
    multiplier_by_sensor = {
        row["sensor_id"]: float(row["travel_time_multiplier"]) for row in latest_metrics
    }
    level_by_sensor = {row["sensor_id"]: row["congestion_level"] for row in latest_metrics}

    graph = defaultdict(list)
    for row in distances_df.collect():
        weight = float(row["distance_meters"]) * multiplier_by_sensor.get(
            row["target_sensor"], 1.0
        )
        graph[row["source_sensor"]].append((row["target_sensor"], weight, row["distance_meters"]))

    queue: list[tuple[float, str, list[str], float]] = [(0.0, source, [source], 0.0)]
    seen: dict[str, float] = {}

    while queue:
        total_cost, node, path, total_distance = heapq.heappop(queue)
        if node == target:
            return {
                "source": source,
                "target": target,
                "evaluated_at": str(latest_timestamp),
                "path": path,
                "estimated_weighted_cost": round(total_cost, 2),
                "distance_meters": round(total_distance, 2),
                "congestion_levels_on_path": [level_by_sensor.get(step, "unknown") for step in path],
            }
        if node in seen and seen[node] <= total_cost:
            continue
        seen[node] = total_cost
        for neighbor, weight, distance in graph.get(node, []):
            heapq.heappush(
                queue,
                (total_cost + weight, neighbor, path + [neighbor], total_distance + distance),
            )

    return {
        "source": source,
        "target": target,
        "evaluated_at": str(latest_timestamp),
        "path": [],
        "error": "No path found between the selected sensors.",
    }
