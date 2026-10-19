from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processedData" / "cleanedTransactions.csv"
OUTPUT_DIR = ROOT / "reports" / "modelReports"
OUTPUT_CSV = OUTPUT_DIR / "inventoryOptimizationRecommendations.csv"
OUTPUT_REPORT = OUTPUT_DIR / "inventoryDemandAudit.txt"

def as_bool(series):
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    usecols = [
        "ProductID", "invoiceDate", "quantity", "unitPrice",
        "isCandidateSale", "isCrossSheetDuplicate"
    ]
    df = pd.read_csv(DATA_PATH, usecols=lambda c: c in usecols, low_memory=False)
    df["invoiceDate"] = pd.to_datetime(df["invoiceDate"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["unitPrice"] = pd.to_numeric(df["unitPrice"], errors="coerce")
    df = df.dropna(subset=["ProductID", "invoiceDate", "quantity", "unitPrice"])
    df = df.loc[(df["quantity"] > 0) & (df["unitPrice"] > 0)].copy()
    if "isCandidateSale" in df:
        df = df.loc[as_bool(df["isCandidateSale"])].copy()
    if "isCrossSheetDuplicate" in df:
        df = df.loc[~as_bool(df["isCrossSheetDuplicate"])].copy()

    df["saleDate"] = df["invoiceDate"].dt.normalize()
    min_date, max_date = df["saleDate"].min(), df["saleDate"].max()
    full_calendar = pd.date_range(min_date, max_date, freq="D")
    recent_start = max_date - pd.Timedelta(days=27)
    long_start = max_date - pd.Timedelta(days=89)

    daily = df.groupby(["ProductID", "saleDate"], as_index=False).agg(
        dailyUnits=("quantity", "sum"),
        transactionRows=("quantity", "size")
    )
    product_meta = df.groupby("ProductID", as_index=False).agg(
        firstSaleDate=("saleDate", "min"),
        lastSaleDate=("saleDate", "max"),
        totalUnitsSold=("quantity", "sum"),
        transactionRows=("quantity", "size")
    )

    records = []
    for row in product_meta.itertuples(index=False):
        product_daily = daily.loc[daily["ProductID"] == row.ProductID, ["saleDate", "dailyUnits"]]
        # Critical fix: use the same 28/90 calendar-day window for every product,
        # instead of truncating the window to a product's first observed sale.
        series = product_daily.set_index("saleDate")["dailyUnits"].reindex(
            full_calendar, fill_value=0.0
        ).astype(float)
        recent = series.loc[series.index >= recent_start]
        long = series.loc[series.index >= long_start]

        recent_rate = float(recent.mean()) if len(recent) else 0.0
        long_rate = float(long.mean()) if len(long) else 0.0
        demand_rate = 0.60 * recent_rate + 0.40 * long_rate
        demand_std = float(long.std(ddof=1)) if len(long) > 1 else 0.0
        if not np.isfinite(demand_std):
            demand_std = 0.0

        lead_time = 14
        review_period = 7
        z = 1.645
        safety_stock = z * demand_std * np.sqrt(lead_time)
        reorder_point = demand_rate * lead_time + safety_stock
        target_stock = demand_rate * (lead_time + review_period) + safety_stock
        days_since_last = int((max_date - row.lastSaleDate).days)
        active_28 = int((recent > 0).sum())
        active_90 = int((long > 0).sum())

        if days_since_last <= 30 and active_28 > 0:
            status = "Active - review replenishment"
        elif days_since_last <= 90 and active_90 > 0:
            status = "Intermittent / slowing"
        else:
            status = "No recent sales - review before ordering"

        records.append({
            "ProductID": row.ProductID,
            "firstSaleDate": row.firstSaleDate.date().isoformat(),
            "lastSaleDate": row.lastSaleDate.date().isoformat(),
            "daysSinceLastSale": days_since_last,
            "transactionRows": int(row.transactionRows),
            "totalUnitsSoldHistorical": float(row.totalUnitsSold),
            "unitsSoldLast28Days": float(recent.sum()),
            "unitsSoldLast90Days": float(long.sum()),
            "activeSalesDaysLast28": active_28,
            "activeSalesDaysLast90": active_90,
            "averageDailyDemandLast28": recent_rate,
            "averageDailyDemandLast90": long_rate,
            "estimatedDailyDemand": demand_rate,
            "demandStdDevDailyLast90": demand_std,
            "estimatedDemandNext30Days": demand_rate * 30,
            "assumedLeadTimeDays": lead_time,
            "assumedReviewPeriodDays": review_period,
            "assumedServiceLevel": 0.95,
            "estimatedSafetyStockUnits": safety_stock,
            "estimatedReorderPointUnits": reorder_point,
            "estimatedTargetStockUnits": target_stock,
            "estimatedReplenishmentUnitsIfStockZero": max(0.0, target_stock),
            "recommendationStatus": status,
            "stockOnHandKnown": False,
            "unitCostKnown": False,
        })

    result = pd.DataFrame(records)
    numeric = result.select_dtypes(include=["number"]).columns
    result[numeric] = result[numeric].round(3)
    result = result.sort_values(
        ["recommendationStatus", "estimatedReplenishmentUnitsIfStockZero"],
        ascending=[True, False]
    )
    result.to_csv(OUTPUT_CSV, index=False)

    top = result.nlargest(10, "estimatedDemandNext30Days")[
        ["ProductID", "unitsSoldLast28Days", "averageDailyDemandLast28",
         "averageDailyDemandLast90", "estimatedDailyDemand",
         "estimatedDemandNext30Days", "activeSalesDaysLast28"]
    ]
    lines = [
        "RETAILPULSE INVENTORY DEMAND AUDIT",
        "=" * 36,
        f"Dataset date range: {min_date.date()} to {max_date.date()}",
        f"Global calendar days: {len(full_calendar)}",
        f"Products analyzed: {result['ProductID'].nunique()}",
        "Demand rate formula: 0.60 * mean daily units over global last 28 days",
        "                   + 0.40 * mean daily units over global last 90 days",
        "Zero-sale calendar days are included in both windows.",
        "",
        "TOP 10 PRODUCTS BY ESTIMATED 30-DAY DEMAND",
        top.to_string(index=False),
        "",
        "LIMITATIONS",
        "- These are historical-rate estimates, not SKU-level Prophet/LSTM forecasts.",
        "- Stock on hand and unit cost are unavailable; do not interpret target stock as a purchase order.",
        "- Lead time 14 days, review period 7 days, service level approximately 95% are assumptions.",
        "- Dataset is historical and ends 2011-12-09.",
    ]
    OUTPUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Inventory demand audit and corrected recommendations completed.")
    print(f"Dataset range: {min_date.date()} to {max_date.date()}")
    print(f"Products: {result['ProductID'].nunique():,}")
    print(f"Corrected recommendations: {OUTPUT_CSV}")
    print(f"Audit report: {OUTPUT_REPORT}")
    print("\nTop 10 by estimated 30-day demand:")
    print(top.to_string(index=False))
    print("\nReminder: recommendations remain estimates, not live purchase orders.")

if __name__ == "__main__":
    main()
