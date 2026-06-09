# Weather Trend Forecasting

This project is for the PM Accelerator data science technical assessment. It uses the Global Weather Repository dataset to clean weather data, explore trends, forecast future temperature, detect anomalies, and visualize geographical weather patterns.


Source: https://www.pmaccelerator.io/product-manager-certification

## Dataset

The dashboard loads the latest version of the Kaggle dataset with KaggleHub:

https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository/code


## Tech Stack

- Python
- pandas
- numpy
- scikit-learn
- plotly
- streamlit
- joblib
- kagglehub

## Project Structure

```text
data/
  README.md
src/
  preprocessing.py
  modeling.py
  visualization.py
outputs/
  figures/
app.py
report.md
requirements.txt
README.md
```

## Methods

The project includes:

- Data cleaning and preprocessing.
- Missing value handling.
- Outlier capping with the interquartile range method.
- Time-series features from the last updated field.
- Temperature lag and rolling average features.
- Forecasting with multiple regression models.
- A simple ensemble model.
- Anomaly detection with Isolation Forest.
- Feature importance analysis.
- Spatial weather visualization.

## Models

The forecasting models are:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- Simple Ensemble

The model metrics are:

- MAE
- RMSE
- R-squared

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run app.py
```
