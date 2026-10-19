from pathlib import Path
import json
import warnings
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[2]
DAILY_FILE = ROOT / "data" / "processedData" / "dailySales.csv"
OUTPUT_DIR = ROOT / "data" / "processedData"
REPORT_DIR = ROOT / "reports" / "modelReports"
FIGURE_DIR = ROOT / "reports" / "figures"
MODEL_DIR = ROOT / "reports" / "modelReports"
HOLDOUT_DAYS = 30
RANDOM_STATE = 42
MODEL_NAME = "prophetDailySalesBaseline"


def calculateMetrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    absoluteErrors = np.abs(actual - predicted)
    nonZeroMask = actual != 0

    mape = (
        float(np.mean(absoluteErrors[nonZeroMask] / np.abs(actual[nonZeroMask])) * 100)
        if nonZeroMask.any()
        else float("nan")
    )

    totalActual = float(np.sum(np.abs(actual)))
    wape = (
        float(np.sum(absoluteErrors) / totalActual * 100)
        if totalActual > 0
        else float("nan")
    )

    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mapeNonZeroDaysPercent": mape,
        "wapePercent": wape,
        "evaluatedDays": int(len(actual)),
        "nonZeroActualDays": int(nonZeroMask.sum()),
    }


def main():
    for directory in (OUTPUT_DIR, REPORT_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    if not DAILY_FILE.exists():
        raise FileNotFoundError(
            f"Daily sales file not found: {DAILY_FILE}\n"
            "Run src/forecasting/prepareTimeSeries.py first."
        )

    dailyData = pd.read_csv(DAILY_FILE, parse_dates=["date"])
    dailyData = dailyData.rename(columns={"date": "ds", "dailySales": "y"})
    dailyData["y"] = pd.to_numeric(dailyData["y"], errors="coerce")
    dailyData = dailyData.dropna(subset=["ds", "y"]).sort_values("ds")

    if dailyData["ds"].duplicated().any():
        raise ValueError("Daily sales data contains duplicate dates.")

    if (dailyData["y"] < 0).any():
        raise ValueError("Daily sales series contains negative values.")

    if len(dailyData) <= HOLDOUT_DAYS + 60:
        raise ValueError("Not enough observations for training and holdout.")

    # Strict chronological split: no holdout observations are used for fitting.
    trainData = dailyData.iloc[:-HOLDOUT_DAYS].copy()
    testData = dailyData.iloc[-HOLDOUT_DAYS:].copy()

    print(f"Training observations: {len(trainData)}")
    print(f"Holdout observations: {len(testData)}")
    print(f"Training ends: {trainData['ds'].max().date()}")
    print(f"Holdout starts: {testData['ds'].min().date()}")
    print(f"Holdout ends: {testData['ds'].max().date()}")

    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.05,
        interval_width=0.95,
    )

    model.fit(trainData[["ds", "y"]])

    # Predict only the holdout dates for an honest evaluation.
    holdoutDates = testData[["ds"]].copy()
    holdoutForecast = model.predict(holdoutDates)

    predictions = holdoutForecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    predictions = predictions.merge(testData, on="ds", how="left", validate="one_to_one")
    predictions = predictions.rename(columns={"y": "actualSales"})
    predictions["forecastSales"] = predictions["yhat"].clip(lower=0)

    metrics = calculateMetrics(
        predictions["actualSales"],
        predictions["forecastSales"],
    )

    metrics["trainingRows"] = int(len(trainData))
    metrics["holdoutRows"] = int(len(testData))
    metrics["trainingStart"] = str(trainData["ds"].min().date())
    metrics["trainingEnd"] = str(trainData["ds"].max().date())
    metrics["holdoutStart"] = str(testData["ds"].min().date())
    metrics["holdoutEnd"] = str(testData["ds"].max().date())
    metrics["model"] = MODEL_NAME
    metrics["holdoutDaysRequested"] = HOLDOUT_DAYS
    metrics["forecastClippedAtZeroForEvaluation"] = True

    forecastFile = OUTPUT_DIR / "prophetHoldoutForecast.csv"
    metricsFile = REPORT_DIR / "prophetBaselineMetrics.json"
    reportFile = REPORT_DIR / "prophetBaselineReport.txt"
    modelFile = MODEL_DIR / "prophetBaselineModel.json"
    figureFile = FIGURE_DIR / "prophetHoldoutForecast.png"
    componentsFile = FIGURE_DIR / "prophetForecastComponents.png"

    predictions.to_csv(forecastFile, index=False)
    metricsFile.write_text(
        json.dumps(metrics, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    modelFile.write_text(model_to_json(model), encoding="utf-8")

    # Plot the test-period predictions against actual sales.
    figure, axis = plt.subplots(figsize=(13, 6))
    axis.plot(
        trainData["ds"].tail(60),
        trainData["y"].tail(60),
        label="Training sales (last 60 days)",
        linewidth=1,
    )
    axis.plot(
        predictions["ds"],
        predictions["actualSales"],
        label="Actual holdout sales",
        marker="o",
        linewidth=1.5,
    )
    axis.plot(
        predictions["ds"],
        predictions["forecastSales"],
        label="Prophet forecast (clipped at zero)",
        marker="o",
        linewidth=1.5,
    )
    axis.fill_between(
        predictions["ds"],
        predictions["yhat_lower"].clip(lower=0),
        predictions["yhat_upper"].clip(lower=0),
        alpha=0.2,
        label="95% prediction interval (display clipped at zero)",
    )
    axis.set_title("Prophet Baseline: 30-Day Holdout Evaluation")
    axis.set_xlabel("Date")
    axis.set_ylabel("Sales amount (source currency)")
    axis.legend()
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(figureFile, dpi=150, bbox_inches="tight")
    plt.close(figure)

    # Plot Prophet's learned components for diagnostic interpretation.
    componentsFigure = model.plot_components(
        model.predict(model.make_future_dataframe(periods=30, freq="D"))
    )
    componentsFigure.set_size_inches(12, 8)
    componentsFigure.tight_layout()
    componentsFigure.savefig(
        componentsFile, dpi=150, bbox_inches="tight"
    )
    plt.close(componentsFigure)

    reportLines = [
        "RETAILPULSE - PROPHET BASELINE EVALUATION",
        "=" * 55,
        "Evaluation method: chronological holdout",
        f"Training observations: {metrics['trainingRows']}",
        f"Training period: {metrics['trainingStart']} to {metrics['trainingEnd']}",
        f"Holdout observations: {metrics['holdoutRows']}",
        f"Holdout period: {metrics['holdoutStart']} to {metrics['holdoutEnd']}",
        "",
        "METRICS",
        "-" * 55,
        f"MAE: {metrics['mae']:.4f}",
        f"RMSE: {metrics['rmse']:.4f}",
        f"MAPE on non-zero actual days: {metrics['mapeNonZeroDaysPercent']:.4f}%",
        f"WAPE: {metrics['wapePercent']:.4f}%",
        f"Holdout days evaluated: {metrics['evaluatedDays']}",
        f"Holdout days with non-zero actual sales: {metrics['nonZeroActualDays']}",
        "",
        "MODEL SETTINGS",
        "-" * 55,
        "Yearly seasonality: enabled",
        "Weekly seasonality: enabled",
        "Daily seasonality: disabled",
        "Seasonality mode: additive",
        "Changepoint prior scale: 0.05",
        "Prediction interval width: 95%",
        "Negative predictions clipped to zero for reported evaluation metrics.",
        "",
        "INTERPRETATION AND LIMITATIONS",
        "-" * 55,
        "MAPE is calculated only on days with non-zero actual sales; it is not "
        "directly comparable to MAPE calculated on all days.",
        "WAPE is calculated over all holdout days.",
        "Only the final 30 observed calendar days are held out. This is a single "
        "holdout and does not establish performance across multiple seasons.",
        "The dataset ends in December 2011; this is a historical evaluation, "
        "not a current real-world forecast.",
        "The target is eligible positive sales, not net sales after returns.",
        "The 12% MAPE acceptance target is not considered met unless the "
        "reported result supports it under the project's agreed metric definition.",
        "",
        "OUTPUTS",
        "-" * 55,
        f"Holdout forecast: {forecastFile}",
        f"Metrics JSON: {metricsFile}",
        f"Model JSON: {modelFile}",
        f"Forecast plot: {figureFile}",
        f"Components plot: {componentsFile}",
    ]
    reportFile.write_text("\n".join(reportLines) + "\n", encoding="utf-8")

    # Save a local reproducibility record with artifact hashes.
    import hashlib
    from datetime import datetime, timezone

    artifactPaths = [
        forecastFile, metricsFile, reportFile, modelFile,
        figureFile, componentsFile
    ]
    artifactRecords = []
    for artifactPath in artifactPaths:
        if artifactPath.exists():
            artifactRecords.append({
                "path": str(artifactPath.relative_to(ROOT)),
                "sha256": hashlib.sha256(artifactPath.read_bytes()).hexdigest(),
                "sizeBytes": artifactPath.stat().st_size,
            })

    runRecord = {
        "modelName": MODEL_NAME,
        "runType": "prophetBaseline",
        "recordedAtUtc": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "holdoutDays": HOLDOUT_DAYS,
            "yearlySeasonality": True,
            "weeklySeasonality": True,
            "dailySeasonality": False,
            "seasonalityMode": "additive",
            "changepointPriorScale": 0.05,
            "trainingRows": len(trainData),
            "holdoutRows": len(testData),
            "randomState": RANDOM_STATE,
        },
        "metrics": metrics,
        "artifacts": artifactRecords,
    }
    localRunLog = ROOT / "reports" / "modelReports" / "reproducibility" / "localExperimentRuns.jsonl"
    localRunLog.parent.mkdir(parents=True, exist_ok=True)
    with localRunLog.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(runRecord, ensure_ascii=False) + "\n")
    print(f"Local experiment record saved: {localRunLog}")

    print("\nProphet baseline completed.")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAPE (non-zero actual days): {metrics['mapeNonZeroDaysPercent']:.4f}%")
    print(f"WAPE: {metrics['wapePercent']:.4f}%")
    print(f"Metrics saved: {metricsFile}")
    print(f"Report saved: {reportFile}")
if __name__ == "__main__":
    main()
