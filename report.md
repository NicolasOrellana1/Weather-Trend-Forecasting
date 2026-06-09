# Weather Trend Forecasting Report

## PM Accelerator Mission

PM Accelerator mission: to break down financial barriers and achieve educational fairness.

Source used for mission wording: https://www.pmaccelerator.io/product-manager-certification

## Objective

This project analyzes the Global Weather Repository dataset to understand weather trends, forecast next-temperature values, detect anomalies, and compare weather conditions across regions.

## Data Cleaning

- Loaded the latest Kaggle weather dataset directly with KaggleHub using `GlobalWeatherRepository.csv`.
- Cleaned column names so they are consistent and easier to use.
- Parsed the `lastupdated` or `last_updated` column as a datetime field.
- Filled numeric missing values with the median.
- Filled categorical missing values with `Unknown`.
- Capped extreme numeric values using the interquartile range method.

## Exploratory Analysis

The analysis focuses on:

- Temperature trends over time.
- Precipitation trends over time.
- Average temperature by country.
- Geographic differences using latitude and longitude.

## Forecasting

The main forecasting target is next temperature in Celsius. The project creates useful time-series features such as:

- Year, month, day, day of week, and hour.
- Temperature lag features.
- Seven-period rolling temperature average.
- Seven-period rolling precipitation average when precipitation data exists.

The models compared are:

- Linear Regression.
- Random Forest Regressor.
- Gradient Boosting Regressor.
- A simple ensemble using Random Forest and Gradient Boosting predictions.

The evaluation metrics are:

- Mean Absolute Error.
- Root Mean Squared Error.
- R-squared.

## Advanced Analysis

The advanced work includes:

- Anomaly detection with Isolation Forest.
- Feature importance using permutation importance.
- Environmental analysis using air quality columns when available.
- Spatial analysis using latitude and longitude.
- Geographical comparison across countries.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run app.py
```

## Final Insight Summary

This project is designed to be simple, readable, and realistic. It demonstrates the full data science workflow: cleaning, exploration, feature engineering, modeling, evaluation, anomaly detection, visualization, and dashboard reporting.
