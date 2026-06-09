import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


MODEL_FILE_PATH = "outputs/best_temperature_model.joblib"
MAX_MODEL_ROWS = 20000
MAX_IMPORTANCE_ROWS = 3000


def get_model_features(weather_data):
    # Select simple numeric features and remove target leakage columns.
    excluded_columns = {
        "target_next_temperature",
        "temperature_fahrenheit",
        "last_updated_epoch",
    }
    numeric_columns = weather_data.select_dtypes(include=["number"]).columns
    feature_columns = [
        column_name
        for column_name in numeric_columns
        if column_name not in excluded_columns
    ]
    return feature_columns


def create_modeling_data(weather_data):
    # Build the feature matrix and target values for next-temperature prediction.
    model_weather_data = weather_data.dropna(subset=["target_next_temperature"]).copy()
    model_weather_data = model_weather_data.tail(MAX_MODEL_ROWS)
    feature_columns = get_model_features(model_weather_data)
    model_features = model_weather_data[feature_columns]
    target_values = model_weather_data["target_next_temperature"]
    return model_features, target_values, feature_columns


def evaluate_predictions(target_values, predicted_values):
    # Calculate common regression metrics for model comparison.
    return {
        "mae": mean_absolute_error(target_values, predicted_values),
        "rmse": np.sqrt(mean_squared_error(target_values, predicted_values)),
        "r2": r2_score(target_values, predicted_values),
    }


def train_forecasting_models(weather_data):
    # Train several simple forecasting models and compare performance.
    model_features, target_values, feature_columns = create_modeling_data(weather_data)

    if len(model_features) < 20:
        raise ValueError("At least 20 usable rows are needed to train and evaluate models.")

    training_features, testing_features, training_target, testing_target = train_test_split(
        model_features,
        target_values,
        test_size=0.2,
        shuffle=False,
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=60, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    results = []
    trained_models = {}

    for model_name, model in models.items():
        model.fit(training_features, training_target)
        predicted_values = model.predict(testing_features)
        metrics = evaluate_predictions(testing_target, predicted_values)
        results.append({"model": model_name, **metrics})
        trained_models[model_name] = model

    ensemble_predictions = np.mean(
        [
            trained_models["Random Forest"].predict(testing_features),
            trained_models["Gradient Boosting"].predict(testing_features),
        ],
        axis=0,
    )
    ensemble_metrics = evaluate_predictions(testing_target, ensemble_predictions)
    results.append({"model": "Simple Ensemble", **ensemble_metrics})

    results_data = pd.DataFrame(results).sort_values("rmse")
    best_model_name = results_data.iloc[0]["model"]
    best_model = trained_models.get(best_model_name, trained_models["Random Forest"])

    joblib.dump(
        {
            "model": best_model,
            "feature_columns": feature_columns,
            "model_name": best_model_name,
        },
        MODEL_FILE_PATH,
    )

    return results_data, trained_models, feature_columns, testing_features, testing_target


def detect_weather_anomalies(weather_data):
    # Detect unusual weather records with Isolation Forest.
    possible_anomaly_columns = [
        "temperature_celsius",
        "humidity",
        "precip_mm",
        "wind_kph",
        "pressure_mb",
        "air_quality_pm2_5",
        "air_quality_pm10",
    ]
    anomaly_columns = [
        column_name for column_name in possible_anomaly_columns if column_name in weather_data.columns
    ]

    if len(anomaly_columns) < 2:
        return weather_data.assign(is_anomaly=False), anomaly_columns

    anomaly_model = IsolationForest(contamination=0.03, random_state=42)
    anomaly_labels = anomaly_model.fit_predict(weather_data[anomaly_columns])
    weather_data_with_anomalies = weather_data.copy()
    weather_data_with_anomalies["is_anomaly"] = anomaly_labels == -1

    return weather_data_with_anomalies, anomaly_columns


def calculate_feature_importance(model, feature_columns, testing_features, testing_target):
    # Calculate permutation importance for the selected model.
    testing_features = testing_features.tail(MAX_IMPORTANCE_ROWS)
    testing_target = testing_target.tail(MAX_IMPORTANCE_ROWS)
    importance_result = permutation_importance(
        model,
        testing_features,
        testing_target,
        n_repeats=3,
        random_state=42,
        n_jobs=-1,
    )
    importance_data = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": importance_result.importances_mean,
        }
    )
    return importance_data.sort_values("importance", ascending=False)
