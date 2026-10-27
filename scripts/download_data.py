import requests
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_weather_predictor.config import RAW_DATA_PATH

# 1. Historical weather API
url = "https://archive-api.open-meteo.com/v1/archive"

# 2. Bengaluru coordinates and date range
params = {
    "latitude": 12.9716,
    "longitude": 77.5946,
    "start_date": "2024-01-01",
    "end_date": "2025-12-31",
    "daily": [
        "temperature_2m_mean",
        "relative_humidity_2m_mean",
        "pressure_msl_mean",
        "wind_speed_10m_mean",
        "precipitation_sum"
    ],
    "timezone": "Asia/Kolkata"
}

# 3. Download the historical data
response = requests.get(url, params=params, timeout=30)
response.raise_for_status()

data = response.json()

# 4. Convert the daily data into a table
df = pd.DataFrame(data["daily"])

# 5. Give the columns shorter names
df = df.rename(columns={
    "time": "date",
    "temperature_2m_mean": "temperature",
    "relative_humidity_2m_mean": "humidity",
    "pressure_msl_mean": "pressure",
    "wind_speed_10m_mean": "wind_speed",
    "precipitation_sum": "precipitation"
})

# 6. Create our target variable
# 1 = measurable precipitation; 0 = no measurable precipitation
df["rain"] = (df["precipitation"] > 0).astype(int)

# 7. Save the dataset
df.to_csv(RAW_DATA_PATH, index=False)

print("Historical weather data downloaded successfully!")
print(f"Total records: {len(df)}")
print(df.head())
print("\nRain/no-rain distribution:")
print(df["rain"].value_counts())
