from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "weather_data.csv"
TRAINING_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "forecast_training_data.csv"
)
BENCHMARK_RESULTS_PATH = (
    PROJECT_ROOT / "results" / "forecast_benchmark_results.csv"
)
MODEL_PATH = PROJECT_ROOT / "models" / "weather_model.pkl"

RAW_COLUMNS = (
    "date",
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "precipitation",
    "rain",
)

WEATHER_FEATURES = (
    "previous_temperature",
    "previous_humidity",
    "previous_pressure",
    "previous_wind_speed",
)
SEASONAL_FEATURES = WEATHER_FEATURES + ("month_sin", "month_cos")
TARGET_COLUMN = "target_rain"
TRAINING_COLUMNS = ("date", *WEATHER_FEATURES, TARGET_COLUMN)
BENCHMARK_COLUMNS = (
    "date",
    TARGET_COLUMN,
    "forecast_precipitation",
    "forecast_prediction",
    "ai_prediction",
    "baseline_prediction",
)
