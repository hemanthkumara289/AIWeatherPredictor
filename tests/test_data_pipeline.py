import sys
from pathlib import Path

import pandas as pd

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from ai_weather_predictor.config import (
    MODEL_PATH,
    RAW_COLUMNS,
    TRAINING_COLUMNS,
    WEATHER_FEATURES,
    TARGET_COLUMN,
)
from ai_weather_predictor.data_loader import (
    load_raw_weather_data,
    load_training_data,
)
from ai_weather_predictor.model import load_model, predict_rainfall
from ai_weather_predictor.preprocessing import prepare_training_data


def test_raw_dataset_has_required_columns() -> None:
    raw_data = load_raw_weather_data()
    assert set(RAW_COLUMNS).issubset(raw_data.columns)


def test_prepared_data_matches_checked_in_dataset() -> None:
    raw_data = load_raw_weather_data()
    prepared = prepare_training_data(raw_data)
    checked_in = load_training_data()
    checked_in["date"] = pd.to_datetime(checked_in["date"])

    assert tuple(prepared.columns) == TRAINING_COLUMNS
    assert all(feature in prepared.columns for feature in WEATHER_FEATURES)
    assert TARGET_COLUMN in prepared.columns
    assert not prepared.isna().any().any()
    pd.testing.assert_frame_equal(
        prepared.reset_index(drop=True),
        checked_in,
        check_dtype=False,
    )


def test_existing_model_artifact_loads() -> None:
    assert MODEL_PATH.is_file()
    assert load_model() is not None


def test_loaded_model_returns_valid_prediction_and_probability() -> None:
    model = load_model()
    training_data = load_training_data()
    features = training_data.loc[:, list(WEATHER_FEATURES)].head(1)

    prediction, rain_probability = predict_rainfall(model, features)

    assert prediction in (0, 1)
    assert 0.0 <= rain_probability <= 1.0
