from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from prophet import Prophet

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "processedData" / "dailySales.csv"
OUTPUT_FILE = ROOT / "reports" / "modelReports" / "prophetTuningResults.csv"

HOLDOUT_DAYS = 30
VALIDATION_DAYS = 30
N_FOLDS = 3

CONFIGS = [
    {"yearly": False, "changepoint": 0.01, "mode": "additive"},
    {"yearly": False, "changepoint": 0.05, "mode": "additive"},
    {"yearly": False, "changepoint": 0.10, "mode": "additive"},
    {"yearly": True,  "changepoint": 0.01, "mode": "additive"},
    {"yearly": True,  "changepoint": 0.05, "mode": "additive"},
    {"yearly": True,  "changepoint": 0.10, "mode": "additive"},
    {"yearly": True,  "changepoint": 0.05, "mode": "multiplicative"},
]

def score(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.maximum(np.asarray(predicted, dtype=float), 0)
    errors = np.abs(actual - predicted)

    nonzero = actual != 0
    mape = (
        np.mean(errors[nonzero] / np.abs(actual[nonzero])) * 100
        if nonzero.any() else np.nan
    )
    denominator = np.abs(actual).sum()
    wape = errors.sum() / denominator * 100 if denominator else np.nan

    return float(mape), float(wape)

def main():
    data = pd.read_csv(DATA_FILE, parse_dates=["date"])
    data = data.rename(columns={"date": "ds", "dailySales": "y"})
    data["y"] = pd.to_numeric(data["y"], errors="coerce")
    data = data.dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)

    if data["ds"].duplicated().any() or (data["y"] < 0).any():
        raise ValueError("Check for duplicate dates or negative sales.")

    # Exclude the final 30-day holdout from all tuning.
    development = data.iloc[:-HOLDOUT_DAYS].reset_index(drop=True)
    required = N_FOLDS * VALIDATION_DAYS + 60

    if len(development) < required:
        raise ValueError("Not enough data for the configured validation folds.")

    results = []

    for config_id, config in enumerate(CONFIGS, start=1):
        fold_scores = []

        for fold in range(N_FOLDS):
            validation_end = len(development) - (
                N_FOLDS - 1 - fold
            ) * VALIDATION_DAYS
            validation_start = validation_end - VALIDATION_DAYS

            train = development.iloc[:validation_start]
            valid = development.iloc[validation_start:validation_end]

            model = Prophet(
                yearly_seasonality=config["yearly"],
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode=config["mode"],
                changepoint_prior_scale=config["changepoint"],
                interval_width=0.95,
            )
            model.fit(train[["ds", "y"]])

            forecast = model.predict(valid[["ds"]])
            mape, wape = score(valid["y"], forecast["yhat"])

            fold_scores.append((mape, wape))
            print(
                f"Config {config_id}/{len(CONFIGS)}, "
                f"fold {fold + 1}/{N_FOLDS}: "
                f"MAPE={mape:.2f}%, WAPE={wape:.2f}%"
            )

        results.append({
            **config,
            "configId": config_id,
            "meanValidationMAPE": float(np.mean([x[0] for x in fold_scores])),
            "meanValidationWAPE": float(np.mean([x[1] for x in fold_scores])),
            "folds": N_FOLDS,
            "validationDaysPerFold": VALIDATION_DAYS,
        })

    output = pd.DataFrame(results).sort_values("meanValidationWAPE")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False)

    print("\n=== Prophet tuning results (ranked by mean validation WAPE) ===")
    print(output.to_string(index=False))
    print(f"\nSaved results to: {OUTPUT_FILE}")
    print("The original baseline model and final holdout were not overwritten.")

if __name__ == "__main__":
    main()
