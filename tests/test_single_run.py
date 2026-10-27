import requests
import pandas as pd


def main() -> None:
    url = "https://single-runs-api.open-meteo.com/v1/forecast"

    params = {
        "latitude": 12.9716,
        "longitude": 77.5946,
        "run": "2025-08-07T00:00",
        "daily": "precipitation_sum",
        "timezone": "UTC",
        "models": "ecmwf_ifs",
    }

    response = requests.get(url, params=params, timeout=60)

    print("HTTP status:", response.status_code)

    if not response.ok:
        print(response.text)
        raise SystemExit(1)

    data = response.json()

    print("\nResponse keys:", list(data.keys()))
    print("\nDaily units:", data.get("daily_units"))

    daily = pd.DataFrame(data["daily"])

    print("\nDaily forecast:")
    print(daily.to_string(index=False))

    print("\nMissing values:")
    print(daily.isna().sum())


if __name__ == "__main__":
    main()
