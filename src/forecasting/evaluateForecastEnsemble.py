from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "data" / "processedData"
REPORT_DIR = ROOT / "reports" / "modelReports"
FIGURE_DIR = ROOT / "reports" / "figures"

PROPHET_FILE = OUTPUT_DIR / "prophetHoldoutForecast.csv"
LSTM_FILE = OUTPUT_DIR / "lstmHoldoutForecast.csv"

PROPHET_WEIGHT = 0.5
LSTM_WEIGHT = 0.5


def calculateMetrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    absoluteErrors = np.abs(actual - predicted)
    nonZeroMask = actual != 0
    totalActual = np.sum(np.abs(actual))

    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mapeNonZeroDaysPercent": (
            float(np.mean(absoluteErrors[nonZeroMask] / np.abs(actual[nonZeroMask])) * 100)
            if nonZeroMask.any() else None
        ),
        "wapePercent": (
            float(np.sum(absoluteErrors) / totalActual * 100)
            if totalActual > 0 else None
        ),
        "evaluatedDays": int(len(actual)),
        "nonZeroActualDays": int(nonZeroMask.sum()),
    }


def main():
    for directory in (OUTPUT_DIR, REPORT_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    for filePath in (PROPHET_FILE, LSTM_FILE):
        if not filePath.exists():
            raise FileNotFoundError(
                f"Required forecast file is missing: {filePath}"
            )

    prophetData = pd.read_csv(PROPHET_FILE, parse_dates=["ds"])
    lstmData = pd.read_csv(LSTM_FILE, parse_dates=["date"])

    requiredProphetColumns = {"ds", "actualSales", "forecastSales"}
    requiredLstmColumns = {"date", "actualSales", "lstmForecast"}

    if not requiredProphetColumns.issubset(prophetData.columns):
        raise ValueError(
            f"Prophet forecast is missing columns: "
            f"{requiredProphetColumns - set(prophetData.columns)}"
        )

    if not requiredLstmColumns.issubset(lstmData.columns):
        raise ValueError(
            f"LSTM forecast is missing columns: "
            f"{requiredLstmColumns - set(lstmData.columns)}"
        )

    prophetData = prophetData[
        ["ds", "actualSales", "forecastSales"]
    ].rename(columns={"ds": "date", "forecastSales": "prophetForecast"})

    combinedData = prophetData.merge(
        lstmData[["date", "actualSales", "lstmForecast"]],
        on="date",
        how="inner",
        suffixes=("_prophet", "_lstm"),
        validate="one_to_one",
    ).sort_values("date")

    if len(combinedData) != 30:
        raise ValueError(
            f"Expected 30 matching holdout dates, found {len(combinedData)}."
        )

    if not np.allclose(
        combinedData["actualSales_prophet"],
        combinedData["actualSales_lstm"],
        rtol=1e-6,
        atol=1e-4,
    ):
        raise ValueError(
            "Actual sales differ between the model files; comparison is unsafe."
        )

    combinedData["actualSales"] = combinedData["actualSales_prophet"]
    combinedData["ensembleForecast"] = (
        PROPHET_WEIGHT * combinedData["prophetForecast"]
        + LSTM_WEIGHT * combinedData["lstmForecast"]
    ).clip(lower=0)

    modelPredictions = {
        "Prophet": combinedData["prophetForecast"],
        "LSTM": combinedData["lstmForecast"],
        "Prophet-LSTM 50:50 Ensemble": combinedData["ensembleForecast"],
    }

    comparison = {}
    for modelName, predictions in modelPredictions.items():
        comparison[modelName] = calculateMetrics(
            combinedData["actualSales"], predictions
        )

    comparisonData = pd.DataFrame([
        {"model": modelName, **metrics}
        for modelName, metrics in comparison.items()
    ])

    forecastFile = OUTPUT_DIR / "ensembleHoldoutForecast.csv"
    comparisonFile = REPORT_DIR / "forecastModelComparison.csv"
    metricsFile = REPORT_DIR / "forecastEnsembleMetrics.json"
    reportFile = REPORT_DIR / "forecastEnsembleReport.txt"
    figureFile = FIGURE_DIR / "forecastModelComparison.png"

    combinedData[
        [
            "date",
            "actualSales",
            "prophetForecast",
            "lstmForecast",
            "ensembleForecast",
        ]
    ].to_csv(forecastFile, index=False)

    comparisonData.to_csv(comparisonFile, index=False)
    metricsFile.write_text(
        json.dumps(
            {
                "prophetWeight": PROPHET_WEIGHT,
                "lstmWeight": LSTM_WEIGHT,
                "holdoutDays": len(combinedData),
                "holdoutStart": str(combinedData["date"].min().date()),
                "holdoutEnd": str(combinedData["date"].max().date()),
                "metrics": comparison,
                "selectionMethod": "Fixed equal weights; no holdout-based tuning",
            },
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )

    figure, axis = plt.subplots(figsize=(14, 6))
    axis.plot(
        combinedData["date"],
        combinedData["actualSales"],
        label="Actual sales",
        linewidth=2,
        marker="o",
    )
    axis.plot(
        combinedData["date"],
        combinedData["prophetForecast"],
        label="Prophet",
        linewidth=1.5,
    )
    axis.plot(
        combinedData["date"],
        combinedData["lstmForecast"],
        label="LSTM",
        linewidth=1.5,
    )
    axis.plot(
        combinedData["date"],
        combinedData["ensembleForecast"],
        label="50:50 ensemble",
        linewidth=1.8,
        linestyle="--",
    )
    axis.set_title("Forecast Comparison: 30-Day Holdout")
    axis.set_xlabel("Date")
    axis.set_ylabel("Sales amount (source currency)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figureFile, dpi=150, bbox_inches="tight")
    plt.close(figure)

    bestWapeModel = min(
        comparison.items(),
        key=lambda item: item[1]["wapePercent"]
        if item[1]["wapePercent"] is not None else float("inf"),
    )[0]

    reportLines = [
        "RETAILPULSE - FORECAST ENSEMBLE EVALUATION",
        "=" * 55,
        f"Holdout period: {combinedData['date'].min().date()} "
        f"to {combinedData['date'].max().date()}",
        f"Holdout days: {len(combinedData)}",
        f"Ensemble weights: Prophet={PROPHET_WEIGHT:.2f}, LSTM={LSTM_WEIGHT:.2f}",
        "Weights were fixed in advance; they were not selected using holdout metrics.",
        "",
        "MODEL COMPARISON",
        "-" * 55,
    ]

    for modelName, metrics in comparison.items():
        reportLines.extend([
            modelName,
            f"  MAE: {metrics['mae']:.4f}",
            f"  RMSE: {metrics['rmse']:.4f}",
            f"  MAPE on non-zero actual days: "
            f"{metrics['mapeNonZeroDaysPercent']:.4f}%",
            f"  WAPE: {metrics['wapePercent']:.4f}%",
            "",
        ])

    reportLines.extend([
        f"Lowest holdout WAPE: {bestWapeModel}",
        "",
        "LIMITATIONS",
        "-" * 55,
        "This is a single historical holdout and does not establish performance "
        "across multiple seasons.",
        "The equal-weight ensemble may perform worse than either individual model.",
        "The target is eligible positive sales, not net sales after returns.",
        "The 12% MAPE target must be assessed using the agreed metric definition.",
        "",
        "OUTPUTS",
        "-" * 55,
        str(forecastFile),
        str(comparisonFile),
        str(metricsFile),
        str(figureFile),
    ])
    reportFile.write_text("\n".join(reportLines) + "\n", encoding="utf-8")

    # Save a local reproducibility record with artifact hashes.
    import hashlib
    from datetime import datetime, timezone

    artifactPaths = [
        forecastFile, comparisonFile, metricsFile, reportFile, figureFile
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
        "modelName": "prophetLstmEqualWeightEnsemble",
        "runType": "forecastEnsembleComparison",
        "recordedAtUtc": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "ensembleMethod": "fixedWeightedAverage",
            "prophetWeight": PROPHET_WEIGHT,
            "lstmWeight": LSTM_WEIGHT,
            "holdoutDays": len(combinedData),
        },
        "metrics": comparison,
        "artifacts": artifactRecords,
    }
    localRunLog = ROOT / "reports" / "modelReports" / "reproducibility" / "localExperimentRuns.jsonl"
    localRunLog.parent.mkdir(parents=True, exist_ok=True)
    with localRunLog.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(runRecord, ensure_ascii=False) + "\n")
    print(f"Local experiment record saved: {localRunLog}")

    print("\nForecast ensemble comparison completed.")
    print(comparisonData.to_string(index=False))
    print(f"\nReport: {reportFile}")
    print(f"Metrics: {metricsFile}")

if __name__ == "__main__":
    main()
