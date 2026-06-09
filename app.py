import pandas as pd
import streamlit as st

from src.modeling import (
    calculate_feature_importance,
    detect_weather_anomalies,
    train_forecasting_models,
)
from src.preprocessing import prepare_weather_data
from src.visualization import (
    build_anomaly_chart,
    build_country_temperature_chart,
    build_precipitation_trend_chart,
    build_temperature_trend_chart,
    build_weather_map,
)


st.set_page_config(page_title="Weather Trend Forecasting", layout="wide")


@st.cache_data
def load_prepared_data():
    # Load and prepare the dataset once for the dashboard.
    return prepare_weather_data()


@st.cache_data
def run_models(weather_data):
    # Train models once and reuse the results in the dashboard.
    return train_forecasting_models(weather_data)


st.title("Weather Trend Forecasting")
st.caption("PM Accelerator technical assessment project")

st.info(
    "PM Accelerator mission: to break down financial barriers and achieve educational fairness."
)

try:
    weather_data, date_column = load_prepared_data()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()
except ValueError as error:
    st.error(str(error))
    st.stop()
except Exception as error:
    st.error(
        "The Kaggle dataset could not be loaded. "
        "Check your internet connection and Kaggle access, then refresh the app."
    )
    st.exception(error)
    st.stop()

# Let the user filter the dashboard by country when the column exists.
filtered_weather_data = weather_data.copy()
if "country" in weather_data.columns:
    countries = ["All Countries"] + sorted(weather_data["country"].dropna().unique().tolist())
    selected_country = st.sidebar.selectbox("Country", countries)
    if selected_country != "All Countries":
        filtered_weather_data = weather_data[weather_data["country"] == selected_country]

# Show a small summary of the prepared dataset.
first_column, second_column, third_column, fourth_column = st.columns(4)
first_column.metric("Rows", f"{len(filtered_weather_data):,}")
second_column.metric("Columns", f"{filtered_weather_data.shape[1]:,}")
third_column.metric("Average Temperature", f"{filtered_weather_data['temperature_celsius'].mean():.1f} C")
fourth_column.metric("Average Humidity", f"{filtered_weather_data.get('humidity', pd.Series([0])).mean():.1f}%")

st.subheader("Exploratory Data Analysis")

# Display the main temperature and precipitation charts.
temperature_chart = build_temperature_trend_chart(filtered_weather_data, date_column)
st.plotly_chart(temperature_chart, use_container_width=True)

precipitation_chart = build_precipitation_trend_chart(filtered_weather_data, date_column)
if precipitation_chart:
    st.plotly_chart(precipitation_chart, use_container_width=True)

country_temperature_chart = build_country_temperature_chart(filtered_weather_data)
if country_temperature_chart:
    st.plotly_chart(country_temperature_chart, use_container_width=True)

st.subheader("Forecasting Models")

try:
    model_results, trained_models, feature_columns, testing_features, testing_target = run_models(weather_data)
    st.dataframe(model_results, use_container_width=True)

    best_model_name = model_results.iloc[0]["model"]
    best_model = trained_models.get(best_model_name, trained_models["Random Forest"])
    feature_importance_data = calculate_feature_importance(
        best_model,
        feature_columns,
        testing_features,
        testing_target,
    )

    st.write(f"Best model by RMSE: **{best_model_name}**")
    st.bar_chart(feature_importance_data.head(12), x="feature", y="importance")
except ValueError as error:
    st.warning(str(error))

st.subheader("Anomaly Detection")

# Detect unusual weather records and visualize them.
weather_data_with_anomalies, anomaly_columns = detect_weather_anomalies(filtered_weather_data)
st.write(f"Isolation Forest used these columns: {', '.join(anomaly_columns)}")
st.write(f"Detected anomalies: {int(weather_data_with_anomalies['is_anomaly'].sum()):,}")

anomaly_chart = build_anomaly_chart(weather_data_with_anomalies, date_column)
if anomaly_chart:
    st.plotly_chart(anomaly_chart, use_container_width=True)

st.subheader("Spatial Analysis")

# Display a geographical temperature map when location columns exist.
weather_map = build_weather_map(filtered_weather_data)
if weather_map:
    st.plotly_chart(weather_map, use_container_width=True)
else:
    st.warning("Latitude and longitude columns are needed for the map.")

st.subheader("Key Takeaways")
st.write(
    "This project cleans global weather data, studies trends, compares forecasting models, "
    "detects unusual weather records, and explores geographical weather differences."
)
