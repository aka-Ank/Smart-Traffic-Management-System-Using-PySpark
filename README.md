# Real-Time Traffic Data Processing with PySpark

This project implements a distributed traffic analytics pipeline on top of Spark using the `METR-LA` and `PeMS-Bay` datasets.

Implemented features:

- Real-time style traffic data processing from sensor streams
- Congestion detection using a vehicle density proxy derived from speed degradation
- Route optimization using distance graphs and live congestion penalties
- Predictive analysis for short-horizon traffic trends
- Data visualization dashboard with graphs and network views
- Optional anomaly detection for sudden slowdowns
- Dynamic public transport fares, congestion pricing, and smart parking fees
- Scalable processing with PySpark DataFrame APIs and window functions


## Dataset note

The source datasets contain sensor speeds and pairwise road-network distances, but do not include raw vehicle counts, GPS coordinates, parking occupancy, or public transport tap-in records. This project therefore uses practical proxies:

- Density proxy: inferred from drop versus each sensor's free-flow speed
- Dynamic pricing: inferred from congestion intensity and demand score
- Dashboard map substitute: network topology view, since the dataset has no sensor coordinates

## Project structure

- `src/traffic_analytics`: PySpark pipeline and CLI
- `dashboard/app.py`: Streamlit dashboard
- `output/`: generated analytics tables

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Runtime requirements:

- Python 3.10 to 3.13
- Java 17 or newer on `PATH`

Check them with:

```bash
python --version
java -version
```

If Spark fails to start with `JAVA_GATEWAY_EXITED`, verify that:

- `java -version` reports 17 or newer
- the virtualenv was created with Python 3.12 or 3.13, not 3.14

## Run the pipeline

---------------------------------------------------------------------
In place of <user>, write your pc's user name
Example: dataset-root /Users/aka_ank/Downloads/traffic-datasets \
---------------------------------------------------------------------

```bash
traffic-analytics run \
  --dataset-root /Users/<user>/Downloads/traffic-datasets \
  --dataset METR-LA \
  --output-root /Users/<user>/Documents/New\ project/output
```

This writes:

- `traffic_metrics.parquet`
- `congestion_alerts.parquet`
- `traffic_forecast.parquet`
- `pricing_recommendations.parquet`
- `anomalies.parquet`
- `summary.json`

## Optimize a route

```bash
traffic-analytics optimize-route \
  --dataset-root /Users/<user>/Downloads/traffic-datasets \
  --dataset METR-LA \
  --source 773869 \
  --target 717446
```

## Launch the dashboard

```bash
streamlit run /Users/<user>/Documents/New\ project/dashboard/app.py
```

Set `TRAFFIC_OUTPUT_ROOT` if you want the dashboard to read a different output directory.

Screenshots of Flask Project: 

<img width="1688" height="914" alt="Screenshot 2026-04-25 at 10 08 46" src="https://github.com/user-attachments/assets/3c200cd6-57cd-4913-a361-e3a7248ecec2" />

<img width="1580" height="946" alt="Screenshot 2026-04-25 at 10 09 29" src="https://github.com/user-attachments/assets/777b0937-d7f4-4a7c-9c01-2297341c101f" />

<img width="1094" height="964" alt="Screenshot 2026-04-25 at 10 12 18" src="https://github.com/user-attachments/assets/31729d29-d0c6-4195-a5a2-7bd8b932ce01" />
