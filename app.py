import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import sys

from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sklearn.metrics import accuracy_score, f1_score
from sklearn.dummy import DummyClassifier

from ai_weather_predictor.config import TARGET_COLUMN, WEATHER_FEATURES
from ai_weather_predictor.data_loader import (
    load_raw_weather_data,
    load_training_data,
)
from ai_weather_predictor.model import create_model, load_model, predict_rainfall
from ai_weather_predictor.weather_service import fetch_weather as fetch_open_meteo_weather

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Weather Laboratory",
    page_icon="🌦️",
    layout="wide"
)

st.title("🌦️ AI Weather Laboratory")
st.caption("Live weather • Machine learning • Forecast analytics")

# --------------------------------------------------
# CITY CONFIGURATION
# --------------------------------------------------

CITIES = {
    "Bengaluru": (12.9716, 77.5946),
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.6139, 77.2090),
    "Chennai": (13.0827, 80.2707),
    "Hyderabad": (17.3850, 78.4867),
    "Kolkata": (22.5726, 88.3639),
}

with st.sidebar:
    st.header("⚙️ Settings")

    city = st.selectbox(
        "Select city",
        list(CITIES.keys())
    )

    st.info(
        "The rainfall model was trained using "
        "Bengaluru historical data. Its AI predictions "
        "are enabled only for Bengaluru."
    )

    st.markdown("---")
    st.caption("Model: Logistic Regression")
    st.caption("Training data: Historical weather")

latitude, longitude = CITIES[city]

# --------------------------------------------------
# FETCH WEATHER DATA
# --------------------------------------------------

@st.cache_data(ttl=600)
def fetch_weather(latitude, longitude):
    return fetch_open_meteo_weather(
        latitude,
        longitude,
        current_variables=[
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m"
        ],
        daily_variables=[
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_probability_max",
            "precipitation_sum"
        ],
        past_days=1,
        forecast_days=7,
        timeout=20
    )


try:
    weather = fetch_weather(latitude, longitude)
except requests.RequestException as error:
    st.error(f"Unable to fetch weather data: {error}")
    st.stop()

current = weather["current"]
daily = weather["daily"]

# --------------------------------------------------
# LIVE WEATHER
# --------------------------------------------------

st.subheader(f"📍 Current Weather — {city}")

m1, m2, m3 = st.columns(3)

m1.metric(
    "Temperature",
    f"{current['temperature_2m']} °C"
)

m2.metric(
    "Humidity",
    f"{current['relative_humidity_2m']}%"
)

m3.metric(
    "Wind Speed",
    f"{current['wind_speed_10m']} km/h"
)

st.caption(
    f"Weather observation time: {current['time']}"
)

# --------------------------------------------------
# OFFICIAL FORECAST
# --------------------------------------------------

st.markdown("---")
st.subheader("🌤️ Official Weather Forecast")

forecast_df = pd.DataFrame({
    "Date": pd.to_datetime(daily["time"]),
    "Maximum Temperature": daily["temperature_2m_max"],
    "Minimum Temperature": daily["temperature_2m_min"],
    "Rain Probability": daily["precipitation_probability_max"],
    "Precipitation (mm)": daily["precipitation_sum"]
})

# Show the forecast for today and the next six days.
today = pd.Timestamp(current["time"]).date()

forecast_df = forecast_df[
    forecast_df["Date"].dt.date >= today
].head(7)

st.dataframe(
    forecast_df,
    width="stretch",
    hide_index=True
)

fig_forecast = px.line(
    forecast_df,
    x="Date",
    y=[
        "Maximum Temperature",
        "Minimum Temperature"
    ],
    markers=True,
    title="Seven-Day Temperature Forecast"
)

st.plotly_chart(
    fig_forecast,
    width="stretch"
)

# --------------------------------------------------
# LOAD HISTORICAL DATA
# --------------------------------------------------

try:
    historical = load_raw_weather_data()
    historical["date"] = pd.to_datetime(
        historical["date"]
    )

    training_df = load_training_data()
    model = load_model()

except (FileNotFoundError, ValueError) as error:
    st.error(
        f"Could not load project data or model: {error}"
    )
    st.stop()

# --------------------------------------------------
# AI PREDICTION
# --------------------------------------------------

st.markdown("---")
st.subheader("🧠 AI Rainfall Prediction")

if city == "Bengaluru":

    # Obtain yesterday's daily-average measurements.
    # The trained model uses previous-day averages,
    # not instantaneous current weather.

    previous_day = date.today().toordinal() - 1
    previous_date = date.fromordinal(previous_day)

    previous_row = historical[
        historical["date"].dt.date == previous_date
    ]

    if previous_row.empty:
        st.warning(
            "The historical dataset does not contain "
            "yesterday's measurements. The AI prediction "
            "is unavailable until suitable data is obtained."
        )

    else:
        row = previous_row.iloc[-1]

        features = pd.DataFrame([{
            "previous_temperature": row["temperature"],
            "previous_humidity": row["humidity"],
            "previous_pressure": row["pressure"],
            "previous_wind_speed": row["wind_speed"]
        }])

        prediction, rain_probability = predict_rainfall(model, features)

        c1, c2 = st.columns(2)

        with c1:
            if prediction == 1:
                st.error("🌧️ AI Prediction: Rain")
            else:
                st.success("☀️ AI Prediction: No Rain")

        with c2:
            st.metric(
                "Model-estimated rain probability",
                f"{rain_probability:.1%}"
            )

        st.caption(
            "Experimental prediction based on yesterday's "
            "weather. The probability is not yet calibrated "
            "and is not a reliable operational forecast."
        )

else:
    st.info(
        "The official forecast is available for this city, "
        "but AI predictions are disabled because the model "
        "was trained only on Bengaluru data."
    )

# --------------------------------------------------
# HISTORICAL ANALYTICS
# --------------------------------------------------

st.markdown("---")
st.subheader("📊 Historical Weather Analytics")

c1, c2 = st.columns(2)

with c1:
    fig_temp = px.line(
        historical,
        x="date",
        y="temperature",
        title="Historical Daily Mean Temperature",
        labels={
            "date": "Date",
            "temperature": "Temperature (°C)"
        }
    )

    st.plotly_chart(
        fig_temp,
        width="stretch"
    )

with c2:
    fig_rain = px.bar(
        historical,
        x="date",
        y="precipitation",
        title="Historical Daily Precipitation",
        labels={
            "date": "Date",
            "precipitation": "Precipitation (mm)"
        }
    )

    st.plotly_chart(
        fig_rain,
        width="stretch"
    )

# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

st.markdown("---")
st.subheader("🔬 Model Laboratory")

features = list(WEATHER_FEATURES)
X = training_df[features]
y = training_df[TARGET_COLUMN]

split_index = int(len(training_df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Reproduce the chronological evaluation.
evaluation_model = create_model()

evaluation_model.fit(X_train, y_train)

predictions = evaluation_model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

baseline_model = DummyClassifier(strategy="most_frequent")
baseline_model.fit(X_train, y_train)
baseline_predictions = baseline_model.predict(X_test)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions
)

c1, c2, c3 = st.columns(3)

c1.metric(
    "AI Test Accuracy",
    f"{accuracy:.2%}"
)

c2.metric(
    "AI F1-score",
    f"{f1:.2%}"
)

c3.metric(
    "Baseline Accuracy",
    f"{baseline_accuracy:.2%}"
)

comparison_df = pd.DataFrame({
    "Model": ["Logistic Regression", "Majority Baseline"],
    "Accuracy": [accuracy, baseline_accuracy]
})

fig_comparison = px.bar(
    comparison_df,
    x="Model",
    y="Accuracy",
    title="AI Model vs Baseline",
    text=comparison_df["Accuracy"].map(
        lambda value: f"{value:.1%}"
    )
)

fig_comparison.update_yaxes(
    tickformat=".0%",
    range=[0, 1]
)

st.plotly_chart(
    fig_comparison,
    width="stretch"
)

st.warning(
    "Historical test metrics describe performance on the "
    "held-out period, not guaranteed future accuracy. "
    "Compare with the official forecast before drawing "
    "conclusions about forecast quality."
)

st.caption(
    "AI Weather Laboratory | Built with Python, "
    "Streamlit, Plotly, Pandas and scikit-learn"
)
