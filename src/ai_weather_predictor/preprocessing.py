import pandas as pd

from ai_weather_predictor.config import (
    RAW_COLUMNS,
    TARGET_COLUMN,
    TRAINING_COLUMNS,
    WEATHER_FEATURES,
)


def prepare_training_data(weather: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [
        column for column in RAW_COLUMNS if column not in weather.columns
    ]
    if missing_columns:
        raise ValueError(
            "Raw weather data is missing required columns: "
            + ", ".join(missing_columns)
        )

    prepared = weather.copy()
    prepared["date"] = pd.to_datetime(prepared["date"])
    prepared = prepared.sort_values("date").reset_index(drop=True)

    raw_features = ("temperature", "humidity", "pressure", "wind_speed")
    for source, feature in zip(raw_features, WEATHER_FEATURES):
        prepared[feature] = prepared[source].shift(1)

    prepared[TARGET_COLUMN] = prepared["rain"]
    prepared = prepared.dropna(subset=list(WEATHER_FEATURES)).copy()
    return prepared[list(TRAINING_COLUMNS)]
