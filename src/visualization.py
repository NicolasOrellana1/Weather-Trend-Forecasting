import plotly.express as px


def build_temperature_trend_chart(weather_data, date_column):
    # Plot average temperature over time.
    chart_data = weather_data.groupby(date_column, as_index=False)["temperature_celsius"].mean()
    return px.line(
        chart_data,
        x=date_column,
        y="temperature_celsius",
        title="Average Temperature Trend",
    )


def build_precipitation_trend_chart(weather_data, date_column):
    # Plot average precipitation over time.
    if "precip_mm" not in weather_data.columns:
        return None

    chart_data = weather_data.groupby(date_column, as_index=False)["precip_mm"].mean()
    return px.line(
        chart_data,
        x=date_column,
        y="precip_mm",
        title="Average Precipitation Trend",
    )


def build_country_temperature_chart(weather_data):
    # Compare average temperature across countries.
    if "country" not in weather_data.columns:
        return None

    chart_data = (
        weather_data.groupby("country", as_index=False)["temperature_celsius"]
        .mean()
        .sort_values("temperature_celsius", ascending=False)
        .head(15)
    )
    return px.bar(
        chart_data,
        x="country",
        y="temperature_celsius",
        title="Top Countries by Average Temperature",
    )


def build_anomaly_chart(weather_data, date_column):
    # Show normal and unusual weather points over time.
    if "is_anomaly" not in weather_data.columns:
        return None

    return px.scatter(
        weather_data,
        x=date_column,
        y="temperature_celsius",
        color="is_anomaly",
        title="Temperature Anomaly Detection",
    )


def build_weather_map(weather_data):
    # Plot weather locations on a world map.
    required_columns = {"latitude", "longitude", "temperature_celsius"}
    if not required_columns.issubset(set(weather_data.columns)):
        return None

    return px.scatter_geo(
        weather_data.sample(min(len(weather_data), 3000), random_state=42),
        lat="latitude",
        lon="longitude",
        color="temperature_celsius",
        hover_name="location_name" if "location_name" in weather_data.columns else None,
        title="Global Temperature Map",
    )
