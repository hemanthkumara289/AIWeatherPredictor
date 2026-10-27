from pathlib import Path

import pandas as pd

from ai_weather_predictor.config import (
    BENCHMARK_COLUMNS,
    BENCHMARK_RESULTS_PATH,
    RAW_COLUMNS,
    RAW_DATA_PATH,
    TRAINING_COLUMNS,
    TRAINING_DATA_PATH,
)


def load_csv(path: Path, required_columns: tuple[str, ...]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required project data was not found: {path}")

    data = pd.read_csv(path)
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(
            f"{path} is missing required columns: {', '.join(missing_columns)}"
        )
    return data


def load_raw_weather_data() -> pd.DataFrame:
    return load_csv(RAW_DATA_PATH, RAW_COLUMNS)


def load_training_data() -> pd.DataFrame:
    return load_csv(TRAINING_DATA_PATH, TRAINING_COLUMNS)


def load_benchmark_results() -> pd.DataFrame:
    return load_csv(BENCHMARK_RESULTS_PATH, BENCHMARK_COLUMNS)
