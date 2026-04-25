from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pathlib import Path
import matplotlib.pyplot as plt

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

# -------------------------------
# PATH SETUP
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "traffic2.csv"
OUTPUT_DIR = BASE_DIR / "output"
PDF_PATH = OUTPUT_DIR / "test2_report.pdf"

OUTPUT_DIR.mkdir(exist_ok=True)

# -------------------------------
# START SPARK
# -------------------------------
spark = SparkSession.builder.appName("Final Traffic Intelligence").getOrCreate()

df = spark.read.csv(str(DATA_PATH), header=True, inferSchema=True)

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
df = df.withColumn("hour", hour("date_time"))

df = df.withColumn(
    "rush_hour",
    when((col("hour").between(8, 10)) | (col("hour").between(17, 20)), 1).otherwise(0)
)

df = df.withColumn("weather_desc_clean", lower(col("weather_description")))

df = df.withColumn(
    "weather_category",
    when(col("weather_desc_clean").contains("clear"), "Clear")
    .when(col("weather_desc_clean").contains("cloud"), "Cloudy")
    .when(col("weather_desc_clean").contains("rain") | col("weather_desc_clean").contains("drizzle"), "Rainy")
    .when(col("weather_desc_clean").contains("thunder"), "Storm")
    .when(col("weather_desc_clean").contains("snow") | col("weather_desc_clean").contains("sleet"), "Snow")
    .when(
        col("weather_desc_clean").contains("fog") |
        col("weather_desc_clean").contains("mist") |
        col("weather_desc_clean").contains("haze") |
        col("weather_desc_clean").contains("smoke"),
        "Low Visibility"
    )
    .otherwise("Other")
)

df = df.withColumn(
    "weather_severity",
    when(col("weather_category") == "Clear", 0)
    .when(col("weather_category") == "Cloudy", 1)
    .when(col("weather_category") == "Rainy", 2)
    .when(col("weather_category") == "Low Visibility", 3)
    .when(col("weather_category") == "Storm", 4)
    .when(col("weather_category") == "Snow", 4)
    .otherwise(1)
)

# -------------------------------
# MULTI-RISK SYSTEM
# -------------------------------
df = df.withColumn(
    "visibility_risk",
    when(col("visibility_in_miles") < 2, 2)
    .when(col("visibility_in_miles") < 5, 1)
    .otherwise(0)
)

df = df.withColumn(
    "pollution_risk",
    when(col("air_pollution_index") > 150, 2)
    .when(col("air_pollution_index") > 100, 1)
    .otherwise(0)
)

df = df.withColumn(
    "traffic_risk",
    when(col("traffic_volume") > 6000, 2)
    .when(col("traffic_volume") > 3000, 1)
    .otherwise(0)
)

df = df.withColumn(
    "total_risk_score",
    col("weather_severity") +
    col("visibility_risk") +
    col("pollution_risk") +
    col("traffic_risk")
)

# -------------------------------
# FINAL RISK LEVEL
# -------------------------------
df = df.withColumn(
    "risk_level",
    when(col("total_risk_score") >= 6, "Severe")
    .when(col("total_risk_score") >= 4, "High")
    .when(col("total_risk_score") >= 2, "Moderate")
    .otherwise("Low")
)

# -------------------------------
# RISK EXPLANATION
# -------------------------------
df = df.withColumn(
    "risk_reason",
    when(col("risk_level") == "Severe", "High traffic + poor weather/visibility")
    .when(col("risk_level") == "High", "Heavy traffic or bad weather")
    .when(col("risk_level") == "Moderate", "Normal traffic with minor issues")
    .otherwise("Smooth conditions")
)

# -------------------------------
# TRANSITION ANALYSIS
# -------------------------------
window = Window.orderBy("date_time")

df = df.withColumn("prev_risk", lag("risk_level").over(window))

transitions = df.filter(col("risk_level") != col("prev_risk")) \
    .select("date_time", "prev_risk", "risk_level", "traffic_volume", "visibility_in_miles")

# -------------------------------
# TABLE FOR REPORT
# -------------------------------
table_df = df.groupBy("hour", "risk_level").agg(
    avg("traffic_volume").alias("avg_traffic"),
    avg("visibility_in_miles").alias("avg_visibility"),
    avg("weather_severity").alias("avg_weather"),
    avg("total_risk_score").alias("avg_risk")
).orderBy("hour")

# -------------------------------
# SUMMARY INSIGHTS
# -------------------------------
most_risky_hour = df.groupBy("hour").agg(avg("total_risk_score").alias("risk")).orderBy(col("risk").desc()).first()
least_risky_hour = df.groupBy("hour").agg(avg("total_risk_score").alias("risk")).orderBy(col("risk")).first()

# -------------------------------
# CHARTS
# -------------------------------
hourly_pd = df.groupBy("hour").agg(avg("total_risk_score").alias("risk")).orderBy("hour").toPandas()

plt.figure()
plt.plot(hourly_pd["hour"], hourly_pd["risk"], marker='o')
plt.title("Risk vs Hour")
chart1 = OUTPUT_DIR / "risk_hour.png"
plt.savefig(chart1)
plt.close()

# -------------------------------
# PDF REPORT
# -------------------------------
doc = SimpleDocTemplate(str(PDF_PATH))
styles = getSampleStyleSheet()

elements = []

# Title
elements.append(Paragraph("Traffic Risk Intelligence Report", styles['Title']))
elements.append(Spacer(1, 12))

# Summary
elements.append(Paragraph(f"Most Risky Hour: {most_risky_hour['hour']}:00", styles['Normal']))
elements.append(Paragraph(f"Safest Hour: {least_risky_hour['hour']}:00", styles['Normal']))
elements.append(Spacer(1, 12))

# Add chart
elements.append(Image(str(chart1), width=400, height=250))
elements.append(Spacer(1, 20))

# Add transitions (top few)
elements.append(Paragraph("Risk Transitions (Sample):", styles['Heading2']))

trans_pd = transitions.limit(10).toPandas()

for _, row in trans_pd.iterrows():
    elements.append(Paragraph(
        f"{row['date_time']} : {row['prev_risk']} → {row['risk_level']}",
        styles['Normal']
    ))

elements.append(Spacer(1, 20))

# Table summary (text version)
elements.append(Paragraph("Hourly Risk Summary:", styles['Heading2']))

table_pd = table_df.limit(10).toPandas()

for _, row in table_pd.iterrows():
    elements.append(Paragraph(
        f"Hour {row['hour']} | {row['risk_level']} | Traffic: {__builtins__.round(row['avg_traffic'], 2)} | Visibility: {__builtins__.round(row['avg_visibility'],2)}",
        styles['Normal']
    ))

# Build PDF
doc.build(elements)

print(f"Report saved at: {PDF_PATH}")

spark.stop()