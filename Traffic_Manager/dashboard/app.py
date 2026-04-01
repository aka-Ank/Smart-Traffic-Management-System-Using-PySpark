from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


OUTPUT_ROOT = Path(
    os.environ.get(
        "TRAFFIC_OUTPUT_ROOT", "/Users/aka_ank/Documents/New project/output"
    )
)


@st.cache_data
def load_parquet(name: str) -> pd.DataFrame:
    path = OUTPUT_ROOT / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_summary() -> dict:
    path = OUTPUT_ROOT / "summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


st.set_page_config(page_title="Traffic Analytics Dashboard", layout="wide")
st.title("Traffic Analytics Dashboard")
st.caption("PySpark outputs over traffic sensor speed and network distance data.")

summary = load_summary()
metrics_df = load_parquet("traffic_metrics.parquet")
forecast_df = load_parquet("traffic_forecast.parquet")
pricing_df = load_parquet("pricing_recommendations.parquet")
anomalies_df = load_parquet("anomalies.parquet")

if not summary:
    st.warning("Run the pipeline first so the dashboard has output files to read.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Records", f"{summary.get('records_processed', 0):,}")
col2.metric("Congestion Alerts", f"{summary.get('congestion_alerts', 0):,}")
col3.metric("Anomalies", f"{summary.get('anomalies_detected', 0):,}")
col4.metric("Forecast Records", f"{summary.get('forecast_records', 0):,}")

if not metrics_df.empty:
    metrics_df["timestamp"] = pd.to_datetime(metrics_df["timestamp"])
    latest = (
        metrics_df.sort_values("timestamp")
        .groupby("sensor_id", as_index=False)
        .tail(1)
        .sort_values("density_score", ascending=False)
        .head(30)
    )
    fig = px.bar(
        latest,
        x="sensor_id",
        y="density_score",
        color="congestion_level",
        title="Highest Congestion Sensors",
    )
    st.plotly_chart(fig, use_container_width=True)

    network_fig = px.scatter(
        latest.reset_index(drop=True),
        x=latest.reset_index().index,
        y="speed",
        size="density_score",
        color="congestion_level",
        hover_data=["sensor_id", "travel_time_multiplier"],
        title="Network Load View (topology substitute for geographic map)",
    )
    st.plotly_chart(network_fig, use_container_width=True)

if not forecast_df.empty:
    forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"])
    sample_sensor = st.selectbox(
        "Forecast Sensor",
        sorted(forecast_df["sensor_id"].astype(str).unique().tolist()),
        index=0,
    )
    forecast_sensor_df = forecast_df[forecast_df["sensor_id"].astype(str) == sample_sensor]
    line_fig = px.line(
        forecast_sensor_df,
        x="timestamp",
        y=["speed", "forecast_speed"],
        title=f"Observed vs Forecast Speed for Sensor {sample_sensor}",
    )
    st.plotly_chart(line_fig, use_container_width=True)

if not pricing_df.empty:
    pricing_fig = px.box(
        pricing_df.head(5000),
        x="congestion_level",
        y="parking_fee_usd",
        color="congestion_level",
        title="Smart Parking Fee Distribution",
    )
    st.plotly_chart(pricing_fig, use_container_width=True)
    st.dataframe(
        pricing_df.sort_values("congestion_charge_usd", ascending=False).head(20),
        use_container_width=True,
    )

if not anomalies_df.empty:
    st.subheader("Detected Anomalies")
    st.dataframe(anomalies_df.sort_values("timestamp", ascending=False).head(50), use_container_width=True)
