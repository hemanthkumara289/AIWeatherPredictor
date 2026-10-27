import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_weather_predictor.weather_service import fetch_weather

weather = fetch_weather(
    12.9716,
    77.5946,
    current_variables=[
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m"
    ],
    timeout=15,
)

print("🌦️ Bengaluru Weather")
print("--------------------")

current = weather["current"]

print(f"Temperature: {current['temperature_2m']} °C")
print(f"Humidity: {current['relative_humidity_2m']}%")
print(f"Wind Speed: {current['wind_speed_10m']} km/h")