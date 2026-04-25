"""Shared helpers for Spark jobs."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from pyspark.sql import SparkSession


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def configure_java_home() -> None:
    """Prefer a Java runtime compatible with Spark 4."""
    current_java_home = os.environ.get("JAVA_HOME", "")
    if any(version in current_java_home for version in ("17", "21", "25")):
        return

    for version in ("17", "21", "25"):
        try:
            result = subprocess.run(
                ["/usr/libexec/java_home", "-v", version],
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue

        java_home = result.stdout.strip()
        if java_home:
            os.environ["JAVA_HOME"] = java_home
            os.environ["PATH"] = f"{java_home}/bin:{os.environ.get('PATH', '')}"
            break


def create_spark_session(app_name: str) -> SparkSession:
    """Create a local Spark session that works well for small Flask-triggered jobs."""
    configure_java_home()
    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
