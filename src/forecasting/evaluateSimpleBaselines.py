from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processedData" / "dailySales.csv"
OUT = ROOT / "reports" / "modelReports" / "simpleForecastBaselineResults.json"
HOLDOUT_DAYS = 30

def score(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.maximum(np.asarray(predicted, dtype=float), 0)
    errors = np.abs(actual - predicted)
    nonzero = actual != 0
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAPE_nonzero_days_percent": float(
            np.mean(errors[nonzero] / np.abs(actual[nonzero])) * 100
        ) if nonzero.any() else None,
        "WAPE_percent": float(errors.sum() / np.abs(actual).sum() * 100),
    }

def main():
    data = pd.read_csv(DATA, parse_dates=["date"])
    data = data.rename(columns={"date": "ds", "dailySales": "y"})
    data["y"] = pd.to_numeric(data["y"], errors="coerce")
    data = data.dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)

    train = data.iloc[:-HOLDOUT_DAYS].copy()
    test = data.iloc[-HOLDOUT_DAYS:].copy()

    if len(train) < 28:
        raise ValueError("Not enough training history.")

    # Repeat the final observed seven-day pattern.
    seasonal_naive = np.resize(train["y"].to_numpy()[-7:], HOLDOUT_DAYS)

    # Use only information available before the holdout.
    rolling_mean = np.full(HOLDOUT_DAYS, train["y"].tail(7).mean())

    # Estimate each weekday's typical sales from the training period.
    weekday_means = train.groupby(train["ds"].dt.dayofweek)["y"].mean()
    weekday_average = test["ds"].dt.dayofweek.map(weekday_means).to_numpy()

    results = {
        "holdoutStart": str(test["ds"].min().date()),
        "holdoutEnd": str(test["ds"].max().date()),
        "holdoutDays": HOLDOUT_DAYS,
        "models": {
            "seasonalNaive7Day": score(test["y"], seasonal_naive),
            "last7DayMean": score(test["y"], rolling_mean),
            "historicalWeekdayMean": score(test["y"], weekday_average),
        },
        "note": "Historical diagnostic comparison; same final holdout previously used for model comparison."
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nSaved to: {OUT}")

if __name__ == "__main__":
    main()
