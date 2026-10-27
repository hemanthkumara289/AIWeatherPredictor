# AIWeatherPredictor

An experimental machine-learning project for predicting measurable rainfall
in Bengaluru, Karnataka, from historical daily weather observations. It also
provides a Streamlit weather dashboard and scripts for comparing seasonal
models with a majority baseline and archived forecasts.

## Features

- Download daily historical weather from the Open-Meteo Archive API.
- Prepare next-day rainfall examples from the previous day's weather.
- Train a logistic-regression rainfall classifier.
- Compare weather-only and seasonal features with a chronological split.
- Benchmark predictions against archived hourly forecast values.
- View current conditions, official forecasts, historical charts, and
  Bengaluru-only model predictions in Streamlit.

## Technologies

Python, pandas, NumPy, scikit-learn, joblib, requests, Plotly, Streamlit, and
pytest for development tests.

## Project layout

```text
AIWeatherPredictor/
├── app.py
├── data/
│   ├── raw/weather_data.csv
│   └── processed/forecast_training_data.csv
├── docs/data_dictionary.md
├── models/weather_model.pkl
├── results/forecast_benchmark_results.csv
├── scripts/
│   ├── analyze_benchmark.py
│   ├── benchmark_forecast.py
│   ├── compare_models.py
│   ├── download_data.py
│   ├── prepare_data.py
│   └── train_model.py
├── src/ai_weather_predictor/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── model.py
│   ├── preprocessing.py
│   └── weather_service.py
├── tests/
│   ├── test_data_pipeline.py
│   └── test_single_run.py
├── predict_weather.py
├── weather_api.py
├── requirements.txt
└── requirements-dev.txt
```

The checked-in CSV datasets and model are retained so the current application
and benchmark can be inspected without downloading or training anything.

## Setup on Windows PowerShell

Python 3.14.7 and the listed dependency versions were used to validate the
current model artifact. From the project root:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

No API key or other secret is currently required. `.env.example` documents
this; the Open-Meteo endpoints used by the project are public.

## Run the workflow

Run commands from the project root. Script inputs and outputs are resolved
relative to the project, so they do not depend on the shell's working
directory.

Download (replaces the raw dataset with a fresh API response):

```powershell
python scripts\download_data.py
```

Prepare the processed training file:

```powershell
python scripts\prepare_data.py
```

Train and save the main weather-only model. This overwrites
`models/weather_model.pkl`:

```powershell
python scripts\train_model.py
```

Compare weather-only, seasonal, and majority-baseline models:

```powershell
python scripts\compare_models.py
```

Run the archived-forecast benchmark. It contacts Open-Meteo and overwrites
`results/forecast_benchmark_results.csv`:

```powershell
python scripts\benchmark_forecast.py
```

Analyze existing benchmark results:

```powershell
python scripts\analyze_benchmark.py
```

Launch the dashboard:

```powershell
streamlit run app.py
```

The standalone current-weather prediction and API utilities can also be run
with `python predict_weather.py` and `python weather_api.py`.

## Data and features

The checked-in raw dataset contains daily records with temperature, humidity,
pressure, wind speed, precipitation, and a binary `rain` label. The label is
1 for any precipitation greater than 0 mm and 0 otherwise. The processed file
contains the previous day's temperature, humidity, pressure, and wind speed,
with current-day rainfall as the target. The model-comparison and benchmark
scripts derive sine/cosine month features separately.

For detailed column definitions, see
[docs/data_dictionary.md](./docs/data_dictionary.md).

## Evaluation snapshot

The checked-in benchmark contains 146 matched days from 2025-08-08 through
2025-12-31. The following reported metrics are reproduced from that file:

| Method | Accuracy | Balanced accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|
| Seasonal AI | 86.30% | 85.54% | 91.49% | 87.76% | 89.58% |
| Majority baseline | 67.12% | 50.00% | — | — | — |
| Archived weather forecast | 84.25% | 86.14% | 95.18% | 80.61% | 87.29% |

These are preliminary results from one evaluation period; they do not
establish that the seasonal AI is conclusively better. The historical target
data is reanalysis-based and is not guaranteed to represent verified
ground-station observations. The archived forecast comparison aggregates
fixed-lead hourly precipitation values by calendar day and may combine values
from different forecast runs.

The checked-in model artifact is the weather-only model, with the four
previous-day weather features. It was successfully loaded with scikit-learn
1.9.1 during this refactor. The seasonal metrics above are from the
comparison/benchmark workflow, not a claim about the artifact's evaluation.

## Tests

Install the development dependencies and run the offline suite:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
```

The Single Runs API diagnostic is intentionally a separate script because it
requires network access and its response may omit precipitation values. Run it
only when desired:

```powershell
python tests\test_single_run.py
```

## Limitations and future work

- Historical labels are reanalysis-based rather than verified station data.
- The stored benchmark covers one relatively short seasonal period.
- The archived hourly forecast values can represent different forecast runs.
- The Bengaluru rainfall model is experimental and should not be treated as
  an operational forecast.
- Further evaluation should use longer, independently verified data and
  repeated time-based test periods.
