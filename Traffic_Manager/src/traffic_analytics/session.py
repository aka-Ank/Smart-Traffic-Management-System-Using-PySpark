from __future__ import annotations

import re
import subprocess
import sys

from pyspark.sql import SparkSession


def _require_supported_python() -> None:
    if sys.version_info >= (3, 14):
        version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        raise RuntimeError(
            "PySpark in this project requires Python 3.10-3.13. "
            f"Current interpreter is Python {version}. "
            "Create the virtualenv with Python 3.13 or 3.12 and reinstall dependencies."
        )


def _require_supported_java() -> None:
    try:
        result = subprocess.run(
            ["java", "-version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Java is not installed or not on PATH. Install Java 17+ and set JAVA_HOME."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Unable to determine the Java runtime. Ensure Java 17+ is installed and JAVA_HOME is set."
        ) from exc

    version_output = result.stderr or result.stdout
    match = re.search(r'version "(\d+)(?:\.(\d+))?', version_output)
    if not match:
        raise RuntimeError(
            "Unable to parse the Java version. Ensure Java 17+ is installed and JAVA_HOME is set."
        )

    major = int(match.group(1))
    if major == 1 and match.group(2):
        major = int(match.group(2))

    if major < 17:
        raise RuntimeError(
            f"Java {major} detected, but this PySpark build requires Java 17+. "
            "Install Java 17 or newer, update JAVA_HOME, and recreate the shell session."
        )


def build_spark(app_name: str = "TrafficAnalytics") -> SparkSession:
    _require_supported_python()
    _require_supported_java()
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
