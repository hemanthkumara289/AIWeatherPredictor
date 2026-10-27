import pandas as pd
import numpy as np
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
    SEASONAL_FEATURES,
    TARGET_COLUMN,
    WEATHER_FEATURES,
)
from ai_weather_predictor.data_loader import load_training_data

# 1. Load the prepared dataset
df = load_training_data()
df["date"] = pd.to_datetime(df["date"])

# 2. Create seasonal features from the date
month = df["date"].dt.month

df["month_sin"] = np.sin(2 * np.pi * month / 12)
df["month_cos"] = np.cos(2 * np.pi * month / 12)

# 3. Define our original weather features
# 4. Define the improved feature set
weather_features = list(WEATHER_FEATURES)
seasonal_features = list(SEASONAL_FEATURES)
target = TARGET_COLUMN

# 5. Split chronologically: first 80% for training, last 20% for testing
split_index = int(len(df) * 0.8)

train = df.iloc[:split_index]
test = df.iloc[split_index:]

X_train = train[weather_features]
X_test = test[weather_features]

X_train_seasonal = train[seasonal_features]
X_test_seasonal = test[seasonal_features]

y_train = train[target]
y_test = test[target]

# 6. Create a reusable model pipeline
def create_model():
    return Pipeline([
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

# 7. Train the original model
original_model = create_model()
original_model.fit(X_train, y_train)

# 8. Train the seasonal model
seasonal_model = create_model()
seasonal_model.fit(X_train_seasonal, y_train)

# 9. Create a simple majority-class baseline
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

# 10. Evaluate all models on the same unseen dates
models = {
    "Majority baseline": (baseline, X_test),
    "Weather only": (original_model, X_test),
    "Weather + season": (seasonal_model, X_test_seasonal),
}

print("TRAINING PERIOD:", train["date"].min().date(),
      "to", train["date"].max().date())

print("TEST PERIOD:", test["date"].min().date(),
      "to", test["date"].max().date())

print("\nMODEL COMPARISON")
print("-" * 75)

for name, (model, X) in models.items():
    predictions = model.predict(X)

    print(f"\n{name}")
    print(f"Accuracy:          {accuracy_score(y_test, predictions):.4f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_test, predictions):.4f}")
    print(f"Precision:         {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Recall:            {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"F1-score:          {f1_score(y_test, predictions, zero_division=0):.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

print("\nExperiment complete!")