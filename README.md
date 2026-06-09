# Weather Trend Forecasting

This project is for the PM Accelerator data science technical assessment. It uses the Global Weather Repository dataset to clean weather data, explore trends, forecast future temperature, detect anomalies, and visualize geographical weather patterns.

## PM Accelerator Mission

PM Accelerator mission: to break down financial barriers and achieve educational fairness.

Source: https://www.pmaccelerator.io/product-manager-certification

## Dataset

The dashboard loads the latest version of the Kaggle dataset with KaggleHub:

https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository/code

No manual CSV download is required.

The code uses this KaggleHub import style:

```python
import kagglehub
from kagglehub import KaggleDatasetAdapter

weather_data = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "nelgiriyewithana/global-weather-repository",
    "GlobalWeatherRepository.csv",
)
```

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

## Demo Video Script

Use this short structure for the required 1-2 minute video:

1. Show the GitHub repository and explain the project goal.
2. Open the Streamlit dashboard.
3. Show the PM Accelerator mission section.
4. Explain the EDA charts for temperature and precipitation.
5. Show the model comparison table and best model.
6. Show anomaly detection and feature importance.
7. End by explaining the main insights and how to run the project.
