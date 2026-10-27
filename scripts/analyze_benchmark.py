import sys
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_weather_predictor.data_loader import load_benchmark_results

# Load the results from our previous experiment
df = load_benchmark_results()

actual = df["target_rain"].astype(int)
ai_predictions = df["ai_prediction"].astype(int)

print("Matched test days:", len(df))
print("Actual rainy days:", actual.sum())
print("\nAI performance on the matched dates")
print("-" * 45)

print(f"Accuracy:          {accuracy_score(actual, ai_predictions):.4f}")
print(f"Balanced accuracy: {balanced_accuracy_score(actual, ai_predictions):.4f}")
print(f"Precision:         {precision_score(actual, ai_predictions, zero_division=0):.4f}")
print(f"Recall:             {recall_score(actual, ai_predictions, zero_division=0):.4f}")
print(f"F1-score:           {f1_score(actual, ai_predictions, zero_division=0):.4f}")

print("\nARCHIVED FORECAST: THRESHOLD COMPARISON")
print("=" * 55)

thresholds = [0, 0.1, 1.0]

for threshold in thresholds:
    predictions = (
        df["forecast_precipitation"] > threshold
    ).astype(int)

    print(f"\nPredicted rain threshold: > {threshold} mm")
    print(f"Days predicted rainy: {predictions.sum()}")

    print(f"Accuracy:          {accuracy_score(actual, predictions):.4f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(actual, predictions):.4f}")
    print(f"Precision:         {precision_score(actual, predictions, zero_division=0):.4f}")
    print(f"Recall:            {recall_score(actual, predictions, zero_division=0):.4f}")
    print(f"F1-score:           {f1_score(actual, predictions, zero_division=0):.4f}")

print("\nThreshold experiment complete!")