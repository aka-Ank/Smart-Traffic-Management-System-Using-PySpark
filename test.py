"""Traffic analysis report with documented steps, saved charts, and PDF output."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, count, hour, max, min, sum, when


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "traffic1.csv"
OUTPUT_DIR = BASE_DIR / "output"
REPORT_PDF_PATH = OUTPUT_DIR / "traffic_report.pdf"


def print_section(title):
    """Print a labeled heading so console output reads like a report."""
    print(f"\n=== {title} ===")


def ensure_output_dir():
    """Create the output folder for saved charts and reports."""
    OUTPUT_DIR.mkdir(exist_ok=True)


def build_labeled_traffic_dataframe(raw_traffic_df):
    """Add traffic labels and dynamic pricing fields used across the report."""
    traffic_df = raw_traffic_df.withColumn("hour_of_day", hour("DateTime"))

    traffic_df = traffic_df.withColumn(
        "traffic_level",
        when(col("Vehicles") <= 20, "Low Traffic")
        .when((col("Vehicles") > 20) & (col("Vehicles") <= 50), "Medium Traffic")
        .otherwise("High Traffic"),
    )

    traffic_df = traffic_df.withColumn(
        "is_peak_hour",
        when(
            ((col("hour_of_day") >= 8) & (col("hour_of_day") <= 11))
            | ((col("hour_of_day") >= 17) & (col("hour_of_day") <= 20)),
            1,
        ).otherwise(0),
    )

    traffic_df = traffic_df.withColumn(
        "time_band",
        when(col("is_peak_hour") == 1, "Peak Hour").otherwise("Off-Peak"),
    )

    traffic_df = traffic_df.withColumn(
        "dynamic_fare",
        when((col("is_peak_hour") == 1) & (col("Vehicles") > 50), 60)
        .when((col("is_peak_hour") == 1) & (col("Vehicles") > 20), 40)
        .when(col("Vehicles") > 50, 40)
        .when(col("Vehicles") > 20, 20)
        .otherwise(10),
    )

    return traffic_df


def build_hourly_summary(traffic_df):
    """Aggregate traffic and fare by hour."""
    return traffic_df.groupBy("hour_of_day").agg(
        sum("Vehicles").alias("total_vehicle_count"),
        avg("Vehicles").alias("average_vehicle_count"),
        sum("dynamic_fare").alias("total_fare_amount"),
        count("*").alias("record_count"),
    )


def build_junction_summary(traffic_df):
    """Aggregate traffic and fare by junction."""
    return traffic_df.groupBy("Junction").agg(
        sum("Vehicles").alias("total_vehicle_count"),
        avg("Vehicles").alias("average_vehicle_count"),
        sum("dynamic_fare").alias("total_fare_amount"),
        avg("dynamic_fare").alias("average_dynamic_fare"),
        count("*").alias("record_count"),
    )


def build_route_time_summary(traffic_df):
    """Aggregate traffic by combined junction and hour to find the best route/time."""
    return traffic_df.groupBy("Junction", "hour_of_day").agg(
        sum("Vehicles").alias("total_vehicle_count"),
        avg("Vehicles").alias("average_vehicle_count"),
        sum("dynamic_fare").alias("total_fare_amount"),
        count("*").alias("record_count"),
    )


def build_summary_lines(traffic_df, hourly_summary_df, junction_summary_df, route_time_df):
    """Build readable summary lines for both console output and the PDF report."""
    dataset_range_row = traffic_df.agg(
        min("DateTime").alias("dataset_start_time"),
        max("DateTime").alias("dataset_end_time"),
        sum("Vehicles").alias("overall_total_vehicle_count"),
        sum("dynamic_fare").alias("overall_total_fare_amount"),
        count("*").alias("overall_record_count"),
        avg("Vehicles").alias("overall_average_vehicle_count"),
    ).first()

    least_congested_hour_row = hourly_summary_df.orderBy(
        col("average_vehicle_count").asc()
    ).first()
    busiest_hour_row = hourly_summary_df.orderBy(
        col("average_vehicle_count").desc()
    ).first()
    least_congested_junction_row = junction_summary_df.orderBy(
        col("average_vehicle_count").asc()
    ).first()
    busiest_junction_row = junction_summary_df.orderBy(
        col("average_vehicle_count").desc()
    ).first()
    best_route_time_row = route_time_df.orderBy(col("average_vehicle_count").asc()).first()
    worst_route_time_row = route_time_df.orderBy(
        col("average_vehicle_count").desc()
    ).first()
    congestion_summary_row = traffic_df.groupBy("time_band").agg(
        avg("Vehicles").alias("average_vehicle_count"),
        sum("Vehicles").alias("total_vehicle_count"),
    ).orderBy(col("average_vehicle_count").desc()).first()
    most_common_traffic_label_row = traffic_df.groupBy("traffic_level").count().orderBy(
        col("count").desc()
    ).first()

    return [
        f"Dataset start time: {dataset_range_row['dataset_start_time']}",
        f"Dataset end time: {dataset_range_row['dataset_end_time']}",
        (
            "Total time covered: "
            f"{dataset_range_row['dataset_end_time'] - dataset_range_row['dataset_start_time']}"
        ),
        f"Total records analysed: {dataset_range_row['overall_record_count']}",
        f"Total vehicles observed: {dataset_range_row['overall_total_vehicle_count']}",
        f"Total fare collected: {dataset_range_row['overall_total_fare_amount']}",
        (
            "Average vehicles per record: "
            f"{dataset_range_row['overall_average_vehicle_count']:.2f}"
        ),
        (
            f"Best time to travel: {least_congested_hour_row['hour_of_day']}:00 "
            f"with average {least_congested_hour_row['average_vehicle_count']:.2f} vehicles"
        ),
        (
            f"Worst time to travel: {busiest_hour_row['hour_of_day']}:00 "
            f"with average {busiest_hour_row['average_vehicle_count']:.2f} vehicles"
        ),
        (
            f"Best route to travel: Junction {least_congested_junction_row['Junction']} "
            f"with average {least_congested_junction_row['average_vehicle_count']:.2f} vehicles"
        ),
        (
            f"Worst route to travel: Junction {busiest_junction_row['Junction']} "
            f"with average {busiest_junction_row['average_vehicle_count']:.2f} vehicles"
        ),
        (
            f"Best route and time: Junction {best_route_time_row['Junction']} at "
            f"{best_route_time_row['hour_of_day']}:00 with average "
            f"{best_route_time_row['average_vehicle_count']:.2f} vehicles"
        ),
        (
            f"Worst route and time: Junction {worst_route_time_row['Junction']} at "
            f"{worst_route_time_row['hour_of_day']}:00 with average "
            f"{worst_route_time_row['average_vehicle_count']:.2f} vehicles"
        ),
        (
            f"Most congested time band: {congestion_summary_row['time_band']} "
            f"with average {congestion_summary_row['average_vehicle_count']:.2f} vehicles"
        ),
        (
            f"Most common traffic label: {most_common_traffic_label_row['traffic_level']} "
            f"({most_common_traffic_label_row['count']} records)"
        ),
    ]


def print_executive_summary(summary_lines):
    """Print the high-level findings in a clean, readable format."""
    print_section("Executive Summary")
    for line in summary_lines:
        print(line)


def print_detailed_tables(traffic_df, hourly_summary_df, junction_summary_df, route_time_df):
    """Print supporting tables after the summary."""
    print_section("Traffic Labels And Dynamic Fare Preview")
    traffic_df.select(
        col("DateTime").alias("date_time"),
        col("Junction").alias("junction_id"),
        col("hour_of_day"),
        col("Vehicles").alias("vehicle_count"),
        col("traffic_level"),
        col("time_band"),
        col("dynamic_fare"),
    ).show(20, truncate=False)

    print_section("Hourly Traffic Summary")
    hourly_summary_df.orderBy("hour_of_day").show(24, truncate=False)

    print_section("Junction Traffic Summary")
    junction_summary_df.orderBy("Junction").show(truncate=False)

    print_section("Best And Worst Route-Time Combinations")
    route_time_df.orderBy(col("average_vehicle_count").asc()).show(10, truncate=False)

    print_section("Most Congested Route-Time Combinations")
    route_time_df.orderBy(col("average_vehicle_count").desc()).show(10, truncate=False)

    print_section("Congestion Summary By Time Band")
    traffic_df.groupBy("time_band", "traffic_level").agg(
        count("*").alias("record_count"),
        avg("Vehicles").alias("average_vehicle_count"),
        sum("dynamic_fare").alias("total_fare_amount"),
    ).orderBy("time_band", "traffic_level").show(truncate=False)


def save_figure(fig, filename, pdf):
    """Save a chart as both PNG and a page inside the PDF report."""
    output_path = OUTPUT_DIR / filename
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    pdf.savefig(fig)
    plt.close(fig)


def save_report(summary_lines, hourly_summary_df, junction_summary_df, route_time_df):
    """Save a PDF report and all charts to the output folder."""
    hourly_summary_pdf = hourly_summary_df.orderBy("hour_of_day").toPandas()
    junction_summary_pdf = junction_summary_df.orderBy("Junction").toPandas()
    best_route_time_pdf = route_time_df.orderBy("average_vehicle_count").limit(10).toPandas()
    worst_route_time_pdf = route_time_df.orderBy(
        col("average_vehicle_count").desc()
    ).limit(10).toPandas()

    with PdfPages(REPORT_PDF_PATH) as pdf:
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        ax.set_title("Smart Traffic Executive Report", fontsize=16, pad=20)
        ax.text(
            0.02,
            0.96,
            "\n".join(summary_lines),
            va="top",
            ha="left",
            fontsize=11,
            family="monospace",
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            hourly_summary_pdf["hour_of_day"],
            hourly_summary_pdf["total_vehicle_count"],
            marker="o",
        )
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Total Vehicle Count")
        ax.set_title("Traffic Volume by Hour")
        ax.grid(True, alpha=0.3)
        save_figure(fig, "traffic_volume_by_hour.png", pdf)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            hourly_summary_pdf["hour_of_day"],
            hourly_summary_pdf["total_fare_amount"],
            marker="o",
            color="darkgreen",
        )
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Total Fare Amount")
        ax.set_title("Dynamic Pricing Revenue by Hour")
        ax.grid(True, alpha=0.3)
        save_figure(fig, "revenue_by_hour.png", pdf)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(
            junction_summary_pdf["Junction"].astype(str),
            junction_summary_pdf["total_vehicle_count"],
            color="steelblue",
        )
        ax.set_xlabel("Junction ID")
        ax.set_ylabel("Total Vehicle Count")
        ax.set_title("Traffic Volume by Junction")
        save_figure(fig, "traffic_volume_by_junction.png", pdf)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(
            junction_summary_pdf["Junction"].astype(str),
            junction_summary_pdf["total_fare_amount"],
            color="sandybrown",
        )
        ax.set_xlabel("Junction ID")
        ax.set_ylabel("Total Fare Amount")
        ax.set_title("Revenue by Junction")
        save_figure(fig, "revenue_by_junction.png", pdf)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis("off")
        ax.set_title("Best Route-Time Combinations", fontsize=14, pad=12)
        ax.table(
            cellText=best_route_time_pdf.round(2).values,
            colLabels=list(best_route_time_pdf.columns),
            loc="center",
        )
        save_figure(fig, "best_route_time_combinations.png", pdf)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis("off")
        ax.set_title("Worst Route-Time Combinations", fontsize=14, pad=12)
        ax.table(
            cellText=worst_route_time_pdf.round(2).values,
            colLabels=list(worst_route_time_pdf.columns),
            loc="center",
        )
        save_figure(fig, "worst_route_time_combinations.png", pdf)


def main():
    """Run the full traffic report and save outputs."""
    ensure_output_dir()

    spark = SparkSession.builder.appName("Traffic Project").getOrCreate()

    raw_traffic_df = spark.read.csv(str(DATA_PATH), header=True, inferSchema=True)
    traffic_df = build_labeled_traffic_dataframe(raw_traffic_df)

    hourly_summary_df = build_hourly_summary(traffic_df)
    junction_summary_df = build_junction_summary(traffic_df)
    route_time_df = build_route_time_summary(traffic_df)

    summary_lines = build_summary_lines(
        traffic_df, hourly_summary_df, junction_summary_df, route_time_df
    )

    print_executive_summary(summary_lines)
    print_detailed_tables(traffic_df, hourly_summary_df, junction_summary_df, route_time_df)
    save_report(summary_lines, hourly_summary_df, junction_summary_df, route_time_df)

    print_section("Saved Files")
    print(f"Report PDF: {REPORT_PDF_PATH}")
    print(f"Charts folder: {OUTPUT_DIR}")

    spark.stop()


if __name__ == "__main__":
    main()
