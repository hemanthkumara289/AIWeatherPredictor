# Data dictionary

## Raw historical data

`data/raw/weather_data.csv` contains one daily Open-Meteo archive record per
date. Its columns are:

| Column | Meaning |
|---|---|
| `date` | Local calendar date in `Asia/Kolkata` |
| `temperature` | Daily mean temperature in degrees Celsius |
| `humidity` | Daily mean relative humidity in percent |
| `pressure` | Daily mean sea-level pressure in hPa |
| `wind_speed` | Daily mean wind speed in km/h |
| `precipitation` | Daily precipitation sum in mm |
| `rain` | Binary label: 1 when precipitation is greater than 0 mm, otherwise 0 |

These historical target values are reanalysis-based and are not guaranteed to
match verified ground-station observations.

## Prepared training data

`data/processed/forecast_training_data.csv` has one fewer row than the raw
dataset because the first day has no prior-day measurements. Each
`previous_*` feature is the previous day's value; `target_rain` is the current
day's `rain` label. The main trained model uses the four previous-day weather
features.

The model-comparison and benchmark scripts additionally derive `month_sin`
and `month_cos` from the target date to represent the annual season.

## Benchmark results

`results/forecast_benchmark_results.csv` stores actual labels, seasonal AI and
majority-baseline predictions, and the archived forecast precipitation and
binary prediction for dates with matched forecast data.
