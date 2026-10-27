import joblib
import sys
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.dummy import DummyClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_weather_predictor.config import MODEL_PATH, TARGET_COLUMN, WEATHER_FEATURES
from ai_weather_predictor.data_loader import load_training_data
from ai_weather_predictor.model import create_model

# 1. Load the prepared dataset
df = load_training_data()

# 2. Select yesterday's weather measurements
X = df[list(WEATHER_FEATURES)]
y = df[TARGET_COLUMN]

# 3. Split chronologically: first 80% for training
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# 4. Build the model
model = create_model()

# 5. Train using historical examples
model.fit(X_train, y_train)

# 6. Predict rainfall on unseen dates
predictions = model.predict(X_test)

# 7. Evaluate the model
print("\n🌦️ IMPROVED WEATHER AI")
print("=" * 35)

print(f"Total records: {len(df)}")
print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")

print("\nModel Performance")
print("-" * 35)

print(f"Accuracy:  {accuracy_score(y_test, predictions):.2%}")
print(
    f"Precision: {precision_score(y_test, predictions, zero_division=0):.2%}"
)
print(
    f"Recall:    {recall_score(y_test, predictions, zero_division=0):.2%}"
)
print(f"F1-score:  {f1_score(y_test, predictions, zero_division=0):.2%}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["No Rain", "Rain"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

# 8. Compare against a simple majority-class baseline
baseline_model = DummyClassifier(strategy="most_frequent")
baseline_model.fit(X_train, y_train)
baseline_predictions = baseline_model.predict(X_test)

baseline_accuracy = accuracy_score(y_test, baseline_predictions)

print("\nBaseline Comparison")
print("-" * 35)
print(f"Baseline accuracy: {baseline_accuracy:.2%}")
print(f"AI model accuracy: {accuracy_score(y_test, predictions):.2%}")

# 9. Save the trained model
joblib.dump(model, MODEL_PATH)

print(f"\nModel saved to {MODEL_PATH}")