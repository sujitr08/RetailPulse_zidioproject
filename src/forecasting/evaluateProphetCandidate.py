from pathlib import Path
import json
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processedData" / "dailySales.csv"
BASELINE_FORECAST = ROOT / "data" / "processedData" / "prophetHoldoutForecast.csv"
REPORTS = ROOT / "reports" / "modelReports"
HOLDOUT_DAYS = 30

def metrics(actual, predicted):
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
        "WAPE_percent": float(
            errors.sum() / np.abs(actual).sum() * 100
        ) if np.abs(actual).sum() else None,
    }

def main():
    data = pd.read_csv(DATA, parse_dates=["date"])
    data = data.rename(columns={"date": "ds", "dailySales": "y"})
    data["y"] = pd.to_numeric(data["y"], errors="coerce")
    data = data.dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)

    train = data.iloc[:-HOLDOUT_DAYS].copy()
    test = data.iloc[-HOLDOUT_DAYS:].copy()

    # Selected using validation folds, not the final holdout.
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
        interval_width=0.95,
    )
    model.fit(train[["ds", "y"]])
    forecast = model.predict(test[["ds"]])

    candidate = pd.DataFrame({
        "ds": test["ds"],
        "actualSales": test["y"].to_numpy(),
        "forecastSales": np.maximum(forecast["yhat"].to_numpy(), 0),
    })

    baseline = pd.read_csv(BASELINE_FORECAST, parse_dates=["ds"])
    required = {"ds", "actualSales", "forecastSales"}
    if not required.issubset(baseline.columns):
        raise ValueError(f"Baseline forecast must contain {required}")

    baseline = baseline[["ds", "actualSales", "forecastSales"]]
    comparison = candidate.merge(
        baseline, on="ds", how="inner", suffixes=("_candidate", "_baseline"),
        validate="one_to_one"
    )

    if len(comparison) != HOLDOUT_DAYS:
        raise ValueError(
            f"Expected {HOLDOUT_DAYS} matched holdout days, got {len(comparison)}."
        )

    if not np.allclose(
        comparison["actualSales_candidate"],
        comparison["actualSales_baseline"],
        equal_nan=True
    ):
        raise ValueError("Baseline and candidate actuals do not match.")

    result = {
        "holdoutStart": str(test["ds"].min().date()),
        "holdoutEnd": str(test["ds"].max().date()),
        "holdoutDays": len(test),
        "baseline": metrics(
            comparison["actualSales_baseline"],
            comparison["forecastSales_baseline"]
        ),
        "candidate": metrics(
            comparison["actualSales_candidate"],
            comparison["forecastSales_candidate"]
        ),
        "candidateConfig": {
            "yearly_seasonality": True,
            "weekly_seasonality": True,
            "seasonality_mode": "multiplicative",
            "changepoint_prior_scale": 0.05,
        },
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "prophetCandidateHoldoutMetrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    candidate.to_csv(
        ROOT / "data" / "processedData" / "prophetCandidateHoldoutForecast.csv",
        index=False
    )

    print(json.dumps(result, indent=2))
    print("\nSaved candidate results separately; baseline artifacts were preserved.")

if __name__ == "__main__":
    main()
