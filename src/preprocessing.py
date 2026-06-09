import os
from pathlib import Path

os.environ.setdefault("KAGGLEHUB_CACHE", str(Path("data") / "kagglehub_cache"))

import kagglehub
import numpy as np
import pandas as pd
from kagglehub import KaggleDatasetAdapter


KAGGLE_DATASET_NAME = "nelgiriyewithana/global-weather-repository"
KAGGLE_DATASET_FILE_PATH = "GlobalWeatherRepository.csv"


def load_weather_data(file_path=KAGGLE_DATASET_FILE_PATH):
    # Load the latest Kaggle dataset directly into a pandas DataFrame.
    return kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        KAGGLE_DATASET_NAME,
        file_path,
    )


def clean_column_names(weather_data):
    # Make column names consistent and easier to use in code.
    cleaned_weather_data = weather_data.copy()
    cleaned_weather_data.columns = (
        cleaned_weather_data.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace(".", "_", regex=False)
    )
    return cleaned_weather_data


def get_first_available_column(weather_data, possible_columns):
    # Find the first matching column that exists in the dataset.
    for column_name in possible_columns:
        if column_name in weather_data.columns:
            return column_name
    return None


def add_time_features(weather_data):
    # Create date-based features from the last updated timestamp.
    prepared_weather_data = weather_data.copy()
    date_column = get_first_available_column(
        prepared_weather_data, ["lastupdated", "last_updated", "date"]
    )

    if date_column is None:
        raise ValueError("A date column named lastupdated, last_updated, or date is required.")

    prepared_weather_data[date_column] = pd.to_datetime(
        prepared_weather_data[date_column], errors="coerce"
    )
    prepared_weather_data = prepared_weather_data.dropna(subset=[date_column])

    prepared_weather_data["year"] = prepared_weather_data[date_column].dt.year
    prepared_weather_data["month"] = prepared_weather_data[date_column].dt.month
    prepared_weather_data["day"] = prepared_weather_data[date_column].dt.day
    prepared_weather_data["day_of_week"] = prepared_weather_data[date_column].dt.dayofweek
    prepared_weather_data["hour"] = prepared_weather_data[date_column].dt.hour
    prepared_weather_data["season"] = prepared_weather_data["month"].map(get_season)

    return prepared_weather_data, date_column


def get_season(month_number):
    # Convert a month number into a simple season label.
    if month_number in [12, 1, 2]:
        return "Winter"
    if month_number in [3, 4, 5]:
        return "Spring"
    if month_number in [6, 7, 8]:
        return "Summer"
    return "Fall"


def fill_missing_values(weather_data, excluded_columns=None):
    # Fill numeric and categorical missing values with simple stable defaults.
    if excluded_columns is None:
        excluded_columns = []

    prepared_weather_data = weather_data.copy()
    numeric_columns = [
        column_name
        for column_name in prepared_weather_data.select_dtypes(include=["number"]).columns
        if column_name not in excluded_columns
    ]
    category_columns = [
        column_name
        for column_name in prepared_weather_data.select_dtypes(exclude=["number"]).columns
        if column_name not in excluded_columns
    ]

    for column_name in numeric_columns:
        prepared_weather_data[column_name] = prepared_weather_data[column_name].fillna(
            prepared_weather_data[column_name].median()
        )

    for column_name in category_columns:
        if prepared_weather_data[column_name].isna().any():
            prepared_weather_data[column_name] = prepared_weather_data[column_name].fillna("Unknown")

    return prepared_weather_data


def cap_extreme_values(weather_data):
    # Cap numeric outliers with the interquartile range method.
    prepared_weather_data = weather_data.copy()
    numeric_columns = prepared_weather_data.select_dtypes(include=["number"]).columns

    for column_name in numeric_columns:
        first_quartile = prepared_weather_data[column_name].quantile(0.25)
        third_quartile = prepared_weather_data[column_name].quantile(0.75)
        interquartile_range = third_quartile - first_quartile

        if interquartile_range == 0:
            continue

        lower_limit = first_quartile - 1.5 * interquartile_range
        upper_limit = third_quartile + 1.5 * interquartile_range
        prepared_weather_data[column_name] = prepared_weather_data[column_name].clip(
            lower=lower_limit, upper=upper_limit
        )

    return prepared_weather_data


def add_forecasting_features(weather_data, date_column):
    # Add lag and rolling features for temperature forecasting.
    prepared_weather_data = weather_data.copy()
    temperature_column = get_first_available_column(
        prepared_weather_data, ["temperature_celsius", "temp_c", "temperature"]
    )
    precipitation_column = get_first_available_column(
        prepared_weather_data, ["precip_mm", "precipitation_mm", "precipitation"]
    )
    location_column = get_first_available_column(
        prepared_weather_data, ["location_name", "city", "location"]
    )

    if temperature_column is None:
        raise ValueError("A temperature column is required for forecasting.")

    grouping_columns = [location_column] if location_column else []
    sort_columns = grouping_columns + [date_column]
    prepared_weather_data = prepared_weather_data.sort_values(sort_columns)

    if grouping_columns:
        grouped_temperature = prepared_weather_data.groupby(grouping_columns)[temperature_column]
        prepared_weather_data["temperature_lag_1"] = grouped_temperature.shift(1)
        prepared_weather_data["temperature_lag_7"] = grouped_temperature.shift(7)
        prepared_weather_data["temperature_rolling_7"] = grouped_temperature.transform(
            lambda values: values.shift(1).rolling(window=7, min_periods=1).mean()
        )
        prepared_weather_data["target_next_temperature"] = grouped_temperature.shift(-1)
    else:
        prepared_weather_data["temperature_lag_1"] = prepared_weather_data[temperature_column].shift(1)
        prepared_weather_data["temperature_lag_7"] = prepared_weather_data[temperature_column].shift(7)
        prepared_weather_data["temperature_rolling_7"] = (
            prepared_weather_data[temperature_column].shift(1).rolling(window=7, min_periods=1).mean()
        )
        prepared_weather_data["target_next_temperature"] = prepared_weather_data[temperature_column].shift(-1)

    if precipitation_column:
        if grouping_columns:
            grouped_precipitation = prepared_weather_data.groupby(grouping_columns)[precipitation_column]
            prepared_weather_data["precipitation_rolling_7"] = grouped_precipitation.transform(
                lambda values: values.shift(1).rolling(window=7, min_periods=1).mean()
            )
        else:
            prepared_weather_data["precipitation_rolling_7"] = (
                prepared_weather_data[precipitation_column].shift(1).rolling(window=7, min_periods=1).mean()
            )

    return prepared_weather_data


def prepare_weather_data(file_path=KAGGLE_DATASET_FILE_PATH):
    # Run all preprocessing steps in one place.
    raw_weather_data = load_weather_data(file_path)
    cleaned_weather_data = clean_column_names(raw_weather_data)
    weather_data_with_time, date_column = add_time_features(cleaned_weather_data)
    weather_data_with_missing_values_filled = fill_missing_values(weather_data_with_time)
    weather_data_with_capped_values = cap_extreme_values(weather_data_with_missing_values_filled)
    prepared_weather_data = add_forecasting_features(weather_data_with_capped_values, date_column)
    prepared_weather_data = prepared_weather_data.replace([np.inf, -np.inf], np.nan)
    prepared_weather_data = fill_missing_values(
        prepared_weather_data,
        excluded_columns=["target_next_temperature"],
    )

    return prepared_weather_data, date_column
