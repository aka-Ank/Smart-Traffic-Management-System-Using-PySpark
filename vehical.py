# =====================================================
# TRAFFIC ANALYSIS + VISUAL PDF REPORT
# =====================================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from reportlab.platypus import SimpleDocTemplate, Image, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# -------------------------------
# 1. Spark Session
# -------------------------------
spark = SparkSession.builder.appName("Traffic Visual Report").getOrCreate()

# -------------------------------
# 2. Load Data
# -------------------------------
df = spark.read.csv("data/traffic3.csv", header=True, inferSchema=True)

df = df.withColumnRenamed("Traffic Situation", "TrafficSituation")

# -------------------------------
# 3. Convert to Pandas
# -------------------------------
pdf = df.toPandas()

# -------------------------------
# 4. Create Output Folder
# -------------------------------
os.makedirs("output", exist_ok=True)

# -------------------------------
# 5. SUMMARY CALCULATIONS
# -------------------------------
total_cars = pdf["CarCount"].sum()
total_bikes = pdf["BikeCount"].sum()
total_buses = pdf["BusCount"].sum()
total_trucks = pdf["TruckCount"].sum()
total_vehicles = pdf["Total"].sum()

# -------------------------------
# 6. VISUALIZATIONS
# -------------------------------

# --- 1. Vehicle Distribution ---
plt.figure()
vehicle_totals = [total_cars, total_bikes, total_buses, total_trucks]
labels = ["Cars", "Bikes", "Buses", "Trucks"]

sns.barplot(x=labels, y=vehicle_totals)
plt.title("Vehicle Distribution")
plt.savefig("output/vehicle_distribution.png")
plt.close()

# --- 2. Traffic Over Time ---
pdf["Time"] = pd.to_datetime(pdf["Time"], format="%I:%M:%S %p")

plt.figure()

sns.lineplot(x=pdf["Time"], y=pdf["Total"])

# Reduce clutter (important)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(8))  

plt.xticks(rotation=45)

plt.title("Traffic Trend Over Time")
plt.xlabel("Time")
plt.ylabel("Total Vehicles")

plt.tight_layout()
plt.savefig("output/traffic_trend.png")
plt.close()

# --- 3. Vehicle Composition Pie ---
plt.figure()
plt.pie(vehicle_totals, labels=labels, autopct='%1.1f%%')
plt.title("Vehicle Composition")
plt.savefig("output/vehicle_pie.png")
plt.close()

# -------------------------------
# 7. CREATE PDF REPORT
# -------------------------------
file_path = "output/vehical_data.pdf"
doc = SimpleDocTemplate(file_path)

styles = getSampleStyleSheet()
elements = []

# Title
elements.append(Paragraph("Traffic Vehicle Analysis Report", styles['Title']))
elements.append(Spacer(1, 20))

# Summary Text
summary_text = f"""
Total Vehicles: {total_vehicles} <br/>
Total Cars: {total_cars} <br/>
Total Bikes: {total_bikes} <br/>
Total Buses: {total_buses} <br/>
Total Trucks: {total_trucks}
"""
elements.append(Paragraph(summary_text, styles['Normal']))
elements.append(Spacer(1, 20))

# Add Images
elements.append(Paragraph("Vehicle Distribution", styles['Heading2']))
elements.append(Image("output/vehicle_distribution.png", width=400, height=250))

elements.append(Spacer(1, 20))

elements.append(Paragraph("Traffic Trend Over Time", styles['Heading2']))
elements.append(Image("output/traffic_trend.png", width=400, height=250))

elements.append(Spacer(1, 20))

elements.append(Paragraph("Vehicle Composition", styles['Heading2']))
elements.append(Image("output/vehicle_pie.png", width=400, height=250))

# -------------------------------
# 8. BUILD PDF
# -------------------------------
doc.build(elements)

print("Visual PDF saved at: output/vehical_data.pdf")

# -------------------------------
# 9. Stop Spark
# -------------------------------
spark.stop()