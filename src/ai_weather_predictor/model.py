from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from ai_weather_predictor.config import MODEL_PATH


def create_model() -> Pipeline:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )


def load_model(path: Path = MODEL_PATH) -> Pipeline:
    if not path.is_file():
        raise FileNotFoundError(f"Trained model was not found: {path}")
    return joblib.load(path)


def predict_rainfall(
    model: Pipeline,
    features: pd.DataFrame,
) -> tuple[int, float]:
    prediction = int(model.predict(features)[0])
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("The rainfall model does not contain the rain class (1).")

    probabilities = model.predict_proba(features)[0]
    rain_probability = float(probabilities[classes.index(1)])
    return prediction, rain_probability
