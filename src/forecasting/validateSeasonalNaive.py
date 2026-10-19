from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processedData" / "dailySales.csv"
OUT = ROOT / "reports" / "modelReports" / "seasonalNaiveValidation.json"

HORIZON = 30
FOLDS = 3

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
        "WAPE_percent": float(errors.sum() / np.abs(actual).sum() * 100)
        if np.abs(actual).sum() else None,
    }

def main():
    data = pd.read_csv(DATA, parse_dates=["date"])
    data = data.rename(columns={"date": "ds", "dailySales": "y"})
    data["y"] = pd.to_numeric(data["y"], errors="coerce")
    data = data.dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)

    if data["ds"].duplicated().any():
        raise ValueError("Duplicate dates found in dailySales.csv.")
    if len(data) < HORIZON * (FOLDS + 1):
        raise ValueError("Not enough observations for the requested folds.")

    results = []
    for fold in range(FOLDS, 0, -1):
        test_end = len(data) - (fold - 1) * HORIZON
        test_start = test_end - HORIZON
        train = data.iloc[:test_start]
        test = data.iloc[test_start:test_end]

        prediction = np.resize(train["y"].to_numpy()[-7:], len(test))

        results.append({
            "fold": FOLDS - fold + 1,
            "trainEnd": str(train["ds"].max().date()),
            "holdoutStart": str(test["ds"].min().date()),
            "holdoutEnd": str(test["ds"].max().date()),
            "metrics": score(test["y"], prediction),
        })

    metric_names = ["MAE", "RMSE", "MAPE_nonzero_days_percent", "WAPE_percent"]
    summary = {
        metric: float(np.mean([
            row["metrics"][metric] for row in results
            if row["metrics"][metric] is not None
        ]))
        for metric in metric_names
    }

    output = {
        "model": "seasonalNaive7Day",
        "horizonDays": HORIZON,
        "folds": results,
        "meanFoldMetrics": summary,
        "note": "Rolling-origin validation; each prediction uses only the preceding training data."
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"\nSaved to: {OUT}")

if __name__ == "__main__":
    main()
