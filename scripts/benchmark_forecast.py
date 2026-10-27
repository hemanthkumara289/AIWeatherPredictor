import requests
import numpy as np
import pandas as pd
import sys
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_weather_predictor.config import (
    BENCHMARK_RESULTS_PATH,
    SEASONAL_FEATURES,
    TARGET_COLUMN,
    WEATHER_FEATURES,
)
from ai_weather_predictor.data_loader import load_training_data

# --------------------------------------------------
# 1. Load our weather dataset
# --------------------------------------------------

df = load_training_data()
df["date"] = pd.to_datetime(df["date"])

# Add seasonal features
month = df["date"].dt.month

df["month_sin"] = np.sin(2 * np.pi * month / 12)
df["month_cos"] = np.cos(2 * np.pi * month / 12)

weather_features = list(WEATHER_FEATURES)
seasonal_features = list(SEASONAL_FEATURES)
target = TARGET_COLUMN

# --------------------------------------------------
# 2. Split chronologically
# --------------------------------------------------

split_index = int(len(df) * 0.8)

train = df.iloc[:split_index].copy()
test = df.iloc[split_index:].copy()

X_train = train[seasonal_features]
y_train = train[target]

X_test = test[seasonal_features]
y_test = test[target]

print("Training period:",
      train["date"].min().date(),
      "to",
      train["date"].max().date())

print("Testing period:",
      test["date"].min().date(),
      "to",
      test["date"].max().date())

# --------------------------------------------------
# 3. Train our seasonal AI model
# --------------------------------------------------

ai_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
    ),
])

ai_model.fit(X_train, y_train)

ai_predictions = ai_model.predict(X_test)

# --------------------------------------------------
# 4. Create the majority-class baseline
# --------------------------------------------------

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(train[weather_features], y_train)

baseline_predictions = baseline.predict(test[weather_features])

# --------------------------------------------------
# 5. Download archived day-ahead forecasts
# --------------------------------------------------

url = "https://previous-runs-api.open-meteo.com/v1/forecast"

params = {
    "latitude": 12.9716,
    "longitude": 77.5946,
    "start_date": test["date"].min().strftime("%Y-%m-%d"),
    "end_date": test["date"].max().strftime("%Y-%m-%d"),
    "hourly": "precipitation_previous_day1",
    "timezone": "Asia/Kolkata",
}

print("\nDownloading archived weather forecasts...")

response = requests.get(url, params=params, timeout=60)
response.raise_for_status()

forecast_data = response.json()

if "hourly" not in forecast_data:
    raise RuntimeError(
        "The API response did not contain hourly forecast data."
    )

hourly = pd.DataFrame(forecast_data["hourly"])

required_columns = ["time", "precipitation_previous_day1"]

missing_columns = [
    col for col in required_columns if col not in hourly.columns
]

if missing_columns:
    raise RuntimeError(
        f"Missing API columns: {missing_columns}. "
        f"Available columns: {hourly.columns.tolist()}"
    )

hourly["time"] = pd.to_datetime(hourly["time"])
hourly["date"] = hourly["time"].dt.normalize()

# Sum the 24 hourly forecast values for each local calendar day.
forecast_daily = (
    hourly.groupby("date")["precipitation_previous_day1"]
    .agg(lambda values: values.sum(min_count=1))
    .reset_index()
)

forecast_daily = forecast_daily.rename(
    columns={"precipitation_previous_day1": "forecast_precipitation"}
)

# --------------------------------------------------
# 6. Match forecasts with the actual test targets
# --------------------------------------------------

comparison = test[["date", target]].merge(
    forecast_daily,
    on="date",
    how="inner",
    validate="one_to_one",
)

# Exclude dates without a usable forecast.
comparison = comparison.dropna(
    subset=["forecast_precipitation", target]
).copy()

if comparison.empty:
    raise RuntimeError(
        "No matching forecast dates were available. "
        "Check the API response and date ranges."
    )

# Use the same basic measurable-rain definition.
# The threshold is applied to forecast precipitation,
# not to the actual precipitation labels.
comparison["forecast_prediction"] = (
    comparison["forecast_precipitation"] > 0
).astype(int)

# Match AI predictions to the exact dates available in the API.
comparison = comparison.merge(
    pd.DataFrame({
        "date": test["date"],
        "ai_prediction": ai_predictions,
        "baseline_prediction": baseline_predictions,
    }),
    on="date",
    how="left",
    validate="one_to_one",
)

comparison = comparison.dropna(
    subset=["ai_prediction", "baseline_prediction"]
)

y_actual = comparison[target].astype(int)

# --------------------------------------------------
# 7. Compare all three methods
# --------------------------------------------------

models = {
    "Majority baseline": comparison["baseline_prediction"].astype(int),
    "Our seasonal AI": comparison["ai_prediction"].astype(int),
    "Archived weather forecast": comparison["forecast_prediction"].astype(int),
}

print("\nMATCHED FORECAST DAYS:", len(comparison))
print("Evaluation period:",
      comparison["date"].min().date(),
      "to",
      comparison["date"].max().date())

print("\nBENCHMARK RESULTS")
print("=" * 65)

for name, predictions in models.items():
    print(f"\n{name}")

    print(
        "Accuracy:",
        f"{accuracy_score(y_actual, predictions):.4f}"
    )

    print(
        "Balanced accuracy:",
        f"{balanced_accuracy_score(y_actual, predictions):.4f}"
    )

    print(
        "Precision:",
        f"{precision_score(y_actual, predictions, zero_division=0):.4f}"
    )

    print(
        "Recall:",
        f"{recall_score(y_actual, predictions, zero_division=0):.4f}"
    )

    print(
        "F1-score:",
        f"{f1_score(y_actual, predictions, zero_division=0):.4f}"
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_actual, predictions))

# --------------------------------------------------
# 8. Save results for later analysis
# --------------------------------------------------

comparison.to_csv(BENCHMARK_RESULTS_PATH, index=False)

print(f"\nResults saved to {BENCHMARK_RESULTS_PATH}")
print("Benchmark complete!")