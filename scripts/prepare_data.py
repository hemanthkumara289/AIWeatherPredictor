import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_weather_predictor.config import TRAINING_DATA_PATH
from ai_weather_predictor.data_loader import load_raw_weather_data
from ai_weather_predictor.preprocessing import prepare_training_data

df = prepare_training_data(load_raw_weather_data())
df.to_csv(TRAINING_DATA_PATH, index=False)

print("Forecasting dataset prepared!")
print(f"Total records: {len(df)}")
print("\nDataset preview:")
print(df.head())

print("\nTarget distribution:")
print(df["target_rain"].value_counts())
