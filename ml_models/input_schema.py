"""Common input schema used by all three ML models."""

from __future__ import annotations

from typing import Any


COMMON_FIELDS = [
    {"name": "hour", "label": "Hour of day (0-23)", "type": "number", "step": "1"},
    {"name": "junction", "label": "Junction ID", "type": "number", "step": "1"},
    {"name": "car_count", "label": "Car count", "type": "number", "step": "1"},
    {"name": "bike_count", "label": "Bike count", "type": "number", "step": "1"},
    {"name": "bus_count", "label": "Bus count", "type": "number", "step": "1"},
    {"name": "truck_count", "label": "Truck count", "type": "number", "step": "1"},
    {"name": "traffic_volume", "label": "Traffic volume", "type": "number", "step": "1"},
    {
        "name": "air_pollution_index",
        "label": "Air pollution index",
        "type": "number",
        "step": "0.01",
    },
    {"name": "humidity", "label": "Humidity", "type": "number", "step": "0.01"},
    {
        "name": "visibility_in_miles",
        "label": "Visibility in miles",
        "type": "number",
        "step": "0.01",
    },
    {"name": "temperature", "label": "Temperature", "type": "number", "step": "0.01"},
    {"name": "rain_p_h", "label": "Rain per hour", "type": "number", "step": "0.01"},
    {"name": "snow_p_h", "label": "Snow per hour", "type": "number", "step": "0.01"},
    {"name": "weather_type", "label": "Weather type", "type": "text", "step": ""},
    {
        "name": "weather_description",
        "label": "Weather description",
        "type": "text",
        "step": "",
    },
    {"name": "day_of_week", "label": "Day of week", "type": "text", "step": ""},
]


def get_example_input() -> dict[str, Any]:
    """Return a sample input payload that works with all three models."""
    return {
        "hour": 9,
        "junction": 2,
        "car_count": 35,
        "bike_count": 8,
        "bus_count": 3,
        "truck_count": 4,
        "traffic_volume": 5200,
        "air_pollution_index": 132,
        "humidity": 78,
        "visibility_in_miles": 3,
        "temperature": 289.4,
        "rain_p_h": 0.2,
        "snow_p_h": 0.0,
        "weather_type": "Clouds",
        "weather_description": "broken clouds",
        "day_of_week": "Tuesday",
    }


def validate_and_convert_input(form_data) -> tuple[dict[str, Any], list[str]]:
    """Validate user input and convert values to the numeric/text types required by the models."""
    cleaned: dict[str, Any] = {}
    errors: list[str] = []

    numeric_fields = {
        "hour": int,
        "junction": int,
        "car_count": float,
        "bike_count": float,
        "bus_count": float,
        "truck_count": float,
        "traffic_volume": float,
        "air_pollution_index": float,
        "humidity": float,
        "visibility_in_miles": float,
        "temperature": float,
        "rain_p_h": float,
        "snow_p_h": float,
    }

    text_fields = {"weather_type", "weather_description", "day_of_week"}

    for field_name, converter in numeric_fields.items():
        raw_value = str(form_data.get(field_name, "")).strip()
        if raw_value == "":
            errors.append(f"{field_name} is required.")
            continue
        try:
            cleaned[field_name] = converter(raw_value)
        except ValueError:
            errors.append(f"{field_name} must be a valid number.")

    for field_name in text_fields:
        raw_value = str(form_data.get(field_name, "")).strip()
        if not raw_value:
            errors.append(f"{field_name} is required.")
        else:
            cleaned[field_name] = raw_value

    if "hour" in cleaned and not 0 <= cleaned["hour"] <= 23:
        errors.append("hour must be between 0 and 23.")

    for field_name in (
        "car_count",
        "bike_count",
        "bus_count",
        "truck_count",
        "traffic_volume",
        "air_pollution_index",
        "humidity",
        "visibility_in_miles",
        "rain_p_h",
        "snow_p_h",
    ):
        if field_name in cleaned and cleaned[field_name] < 0:
            errors.append(f"{field_name} cannot be negative.")

    return cleaned, errors

