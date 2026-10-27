from collections.abc import Sequence
from typing import Any

import requests

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_weather(
    latitude: float,
    longitude: float,
    *,
    current_variables: Sequence[str],
    daily_variables: Sequence[str] | None = None,
    past_days: int | None = None,
    forecast_days: int | None = None,
    timezone: str = "Asia/Kolkata",
    timeout: int = 20,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "latitude": latitude,
        "longitude": longitude,
        "current": list(current_variables),
        "timezone": timezone,
    }
    if daily_variables is not None:
        params["daily"] = list(daily_variables)
    if past_days is not None:
        params["past_days"] = past_days
    if forecast_days is not None:
        params["forecast_days"] = forecast_days

    response = requests.get(OPEN_METEO_URL, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()
