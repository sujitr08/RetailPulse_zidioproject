from pathlib import Path
import warnings
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf


ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "processedData" / "cleanedTransactions.csv"
OUTPUT_DIR = ROOT / "data" / "processedData"
REPORT_DIR = ROOT / "reports" / "modelReports"
FIGURE_DIR = ROOT / "reports" / "figures"

DAILY_OUTPUT = OUTPUT_DIR / "dailySales.csv"
REPORT_FILE = REPORT_DIR / "timeSeriesPreparationReport.txt"

# Weekly seasonality is a reasonable first decomposition for daily retail data.
SEASONAL_PERIOD = 7


def main():
    for directory in (OUTPUT_DIR, REPORT_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_FILE}\n"
            "Run src/dataProcessing/cleanData.py first."
        )

    print("Loading cleaned transactions...")
    transactions = pd.read_csv(
        DATA_FILE,
        usecols=["invoiceDate", "SalesAmount", "isCandidateSale"],
        low_memory=False,
    )

    transactions["invoiceDate"] = pd.to_datetime(
        transactions["invoiceDate"], errors="coerce"
    )
    transactions["SalesAmount"] = pd.to_numeric(
        transactions["SalesAmount"], errors="coerce"
    )

    transactions = transactions.dropna(
        subset=["invoiceDate", "SalesAmount", "isCandidateSale"]
    )

    # Read the flag explicitly to avoid treating a text value such as "False"
    # as True.
    if transactions["isCandidateSale"].dtype == object:
        candidateMask = (
            transactions["isCandidateSale"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1", "yes"])
        )
    else:
        candidateMask = transactions["isCandidateSale"].astype(bool)

    candidateSales = transactions.loc[candidateMask].copy()
    candidateSales = candidateSales[
        (candidateSales["SalesAmount"] > 0)
    ]

    if candidateSales.empty:
        raise ValueError(
            "No eligible positive sales found. Check isCandidateSale "
            "and SalesAmount in the cleaned dataset."
        )

    candidateSales["saleDate"] = candidateSales["invoiceDate"].dt.normalize()

    # Sum eligible sales per calendar day.
    dailySales = (
        candidateSales.groupby("saleDate")["SalesAmount"]
        .sum()
        .sort_index()
        .rename("dailySales")
    )

    # Insert zero-sales dates between the first and last observed dates.
    fullDateRange = pd.date_range(
        start=dailySales.index.min(),
        end=dailySales.index.max(),
        freq="D",
    )
    observedDateCount = len(dailySales)
    dailySales = dailySales.reindex(fullDateRange, fill_value=0.0)
    dailySales.index.name = "date"

    dailyFrame = dailySales.reset_index()
    dailyFrame["date"] = dailyFrame["date"].dt.strftime("%Y-%m-%d")
    dailyFrame.to_csv(DAILY_OUTPUT, index=False)

    values = dailySales.astype(float)

    if len(values) < 2 * SEASONAL_PERIOD:
        raise ValueError(
            f"At least {2 * SEASONAL_PERIOD} daily observations are needed "
            "for weekly seasonal decomposition."
        )

    # Augmented Dickey-Fuller stationarity test.
    adfResult = adfuller(values, autolag="AIC")
    adfStatistic = float(adfResult[0])
    adfPValue = float(adfResult[1])
    adfLags = int(adfResult[2])
    adfObservations = int(adfResult[3])

    stationarityConclusion = (
        "Evidence against a unit root at the 5% level."
        if adfPValue < 0.05
        else "Insufficient evidence to reject a unit root at the 5% level."
    )

    # Weekly decomposition: trend, seasonal component and residual.
    decomposition = seasonal_decompose(
        values,
        model="additive",
        period=SEASONAL_PERIOD,
        extrapolate_trend="freq",
    )

    decompositionFigure = decomposition.plot()
    decompositionFigure.set_size_inches(12, 9)
    decompositionFigure.suptitle(
        "Daily Retail Sales: Weekly Seasonal Decomposition", y=1.01
    )
    decompositionFigure.tight_layout()
    decompositionPath = FIGURE_DIR / "salesTimeSeriesDecomposition.png"
    decompositionFigure.savefig(decompositionPath, dpi=150, bbox_inches="tight")
    plt.close(decompositionFigure)

    # Daily sales overview.
    salesFigure, salesAxis = plt.subplots(figsize=(13, 5))
    salesAxis.plot(values.index, values.values, linewidth=0.8)
    salesAxis.set_title("Daily Eligible Sales")
    salesAxis.set_xlabel("Date")
    salesAxis.set_ylabel("Sales amount (source currency)")
    salesAxis.grid(True, alpha=0.3)
    salesFigure.tight_layout()
    salesPath = FIGURE_DIR / "dailySalesTimeSeries.png"
    salesFigure.savefig(salesPath, dpi=150, bbox_inches="tight")
    plt.close(salesFigure)

    # Autocorrelation diagnostic, up to 30 lags where data permits.
    acfFigure, acfAxis = plt.subplots(figsize=(10, 5))
    plot_acf(
        values,
        lags=min(30, len(values) // 2 - 1),
        ax=acfAxis,
        zero=False,
    )
    acfAxis.set_title("Daily Sales Autocorrelation")
    acfFigure.tight_layout()
    acfPath = FIGURE_DIR / "dailySalesAutocorrelation.png"
    acfFigure.savefig(acfPath, dpi=150, bbox_inches="tight")
    plt.close(acfFigure)

    reportLines = [
        "RETAILPULSE - TIME SERIES PREPARATION REPORT",
        "=" * 55,
        f"Input file: {DATA_FILE}",
        "Target: sum of positive SalesAmount for rows marked isCandidateSale",
        "Interpretation: eligible gross sales, not net sales after returns",
        "Currency: source dataset currency; not independently verified",
        "",
        "DATA COVERAGE",
        "-" * 55,
        f"Start date: {values.index.min().date()}",
        f"End date: {values.index.max().date()}",
        f"Calendar-day observations: {len(values)}",
        f"Days with observed eligible sales before date filling: {observedDateCount}",
        f"Inserted zero-sales calendar days: {len(values) - observedDateCount}",
        f"Candidate transaction rows used: {len(candidateSales):,}",
        f"Total eligible sales amount: {values.sum():,.2f}",
        f"Mean daily sales: {values.mean():,.2f}",
        f"Median daily sales: {values.median():,.2f}",
        f"Standard deviation: {values.std():,.2f}",
        f"Minimum daily sales: {values.min():,.2f}",
        f"Maximum daily sales: {values.max():,.2f}",
        "",
        "STATIONARITY: AUGMENTED DICKEY-FULLER TEST",
        "-" * 55,
        f"ADF statistic: {adfStatistic:.6f}",
        f"p-value: {adfPValue:.8f}",
        f"Used lags: {adfLags}",
        f"Observations used: {adfObservations}",
        f"5% critical value: {adfResult[4]['5%']:.6f}",
        f"Conclusion: {stationarityConclusion}",
        "Note: this test evaluates the level series; it does not prove "
        "forecast quality or rule out all forms of seasonality.",
        "",
        "SEASONAL DECOMPOSITION",
        "-" * 55,
        f"Model: additive",
        f"Seasonal period: {SEASONAL_PERIOD} days (weekly)",
        "Interpretation: inspect the trend, weekly pattern and residual "
        "components in the saved decomposition figure.",
        "",
        "OUTPUTS",
        "-" * 55,
        f"Daily series: {DAILY_OUTPUT}",
        f"Daily sales plot: {salesPath}",
        f"Decomposition plot: {decompositionPath}",
        f"Autocorrelation plot: {acfPath}",
        "",
        "LIMITATIONS",
        "-" * 55,
        "Zero-filled dates mean no eligible sales were recorded on those dates; "
        "they do not establish that the business was open.",
        "The sales target excludes rows not marked as candidate sales and "
        "therefore is not net revenue after returns or cancellations.",
        "The dataset is historical and ends at its last recorded transaction.",
        "Forecast accuracy must be measured separately on a chronological holdout.",
    ]

    REPORT_FILE.write_text("\n".join(reportLines) + "\n", encoding="utf-8")

    print("\nTime-series preparation completed.")
    print(f"Daily observations: {len(values)}")
    print(f"Date range: {values.index.min().date()} to {values.index.max().date()}")
    print(f"ADF p-value: {adfPValue:.8f}")
    print(f"Stationarity conclusion: {stationarityConclusion}")
    print(f"Daily series saved: {DAILY_OUTPUT}")
    print(f"Report saved: {REPORT_FILE}")
    print(f"Figures saved: {FIGURE_DIR}")

if __name__ == "__main__":
    main()
