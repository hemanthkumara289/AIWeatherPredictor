import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_weather_predictor.model import load_model, predict_rainfall
from ai_weather_predictor.weather_service import fetch_weather

# 1. Load our trained AI model
model = load_model()

# 2. Bengaluru coordinates
weather = fetch_weather(
    12.9716,
    77.5946,
    current_variables=[
        "temperature_2m",
        "relative_humidity_2m",
        "surface_pressure",
        "wind_speed_10m"
    ],
    timeout=15,
)
current = weather["current"]

# 4. Prepare the data for our AI
features = pd.DataFrame([{
    "temperature": current["temperature_2m"],
    "humidity": current["relative_humidity_2m"],
    "pressure": current["surface_pressure"],
    "wind_speed": current["wind_speed_10m"]
}])

# 5. Ask the AI to predict
# 6. Get the model's estimated probability
prediction, rain_probability = predict_rainfall(model, features)

# 7. Display results
print("\n🌦️ AI WEATHER PREDICTION")
print("=" * 30)

print("Location: Bengaluru")
print(f"Temperature: {current['temperature_2m']} °C")
print(f"Humidity: {current['relative_humidity_2m']}%")
print(f"Wind Speed: {current['wind_speed_10m']} km/h")
print(f"Pressure: {current['surface_pressure']} hPa")

print("\n🧠 AI RESULT")

if prediction == 1:
    print("Prediction: Rain likely 🌧️")
else:
    print("Prediction: No measurable rain predicted ☀️")

print(f"Model-estimated rain probability: {rain_probability:.1%}")

print("\nNote: This is an experimental model,")
print("not a reliable operational weather forecast.")
