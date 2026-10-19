from pathlib import Path
import json
import math
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processedData" / "cleanedTransactions.csv"
REPORT_DIR = ROOT / "reports" / "modelReports"
FIGURE_DIR = ROOT / "reports" / "figures"

# Explicit, editable planning assumptions; replace with supplier/business data when available.
LEAD_TIME_DAYS = 14
REVIEW_PERIOD_DAYS = 7
PLANNING_HORIZON_DAYS = 30
SERVICE_LEVEL_Z = 1.645  # Approx. 95% one-sided service level
RECENT_WINDOW_DAYS = 28
LONG_WINDOW_DAYS = 90
MIN_ACTIVE_DAYS_FOR_RATE = 7


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Required input not found: {DATA_PATH}\n"
            "Run the data-cleaning pipeline first."
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    usecols = [
        "ProductID", "invoiceDate", "quantity", "unitPrice",
        "isCandidateSale", "isCrossSheetDuplicate", "SalesAmount",
        "StockQuantity", "UnitCost"
    ]
    # Load only the fields needed for this module to reduce memory use.
    df = pd.read_csv(DATA_PATH, usecols=lambda c: c in usecols, low_memory=False)

    required = {"ProductID", "invoiceDate", "quantity", "unitPrice"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in cleanedTransactions.csv: {sorted(missing)}")

    df["invoiceDate"] = pd.to_datetime(df["invoiceDate"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["unitPrice"] = pd.to_numeric(df["unitPrice"], errors="coerce")
    df = df.dropna(subset=["ProductID", "invoiceDate", "quantity", "unitPrice"])
    df["ProductID"] = df["ProductID"].astype(str)

    if "isCandidateSale" in df.columns:
        candidate = df["isCandidateSale"].astype(str).str.strip().str.lower().isin(
            ["true", "1", "yes"]
        )
        df = df.loc[candidate].copy()
    else:
        df = df.loc[(df["quantity"] > 0) & (df["unitPrice"] > 0)].copy()

    # The workbook sheets overlap; exclude only rows explicitly flagged as older-sheet
    # cross-sheet copies. Do not remove other repeated transactions.
    if "isCrossSheetDuplicate" in df.columns:
        cross_dup = df["isCrossSheetDuplicate"].astype(str).str.strip().str.lower().isin(
            ["true", "1", "yes"]
        )
        df = df.loc[~cross_dup].copy()

    df = df.loc[(df["quantity"] > 0) & (df["unitPrice"] > 0)].copy()
    if df.empty:
        raise ValueError("No qualifying positive-quantity, positive-price product sales remain.")

    df["saleDate"] = df["invoiceDate"].dt.normalize()
    max_date = df["saleDate"].max()
    min_date = df["saleDate"].min()
    recent_start = max_date - pd.Timedelta(days=RECENT_WINDOW_DAYS - 1)
    long_start = max_date - pd.Timedelta(days=LONG_WINDOW_DAYS - 1)
    recent30_start = max_date - pd.Timedelta(days=29)

    # Aggregate product-day quantities; duplicate invoice lines within the source are
    # retained unless they were explicitly flagged as cross-sheet copies.
    daily = (
        df.groupby(["ProductID", "saleDate"], as_index=False)
          .agg(dailyUnits=("quantity", "sum"),
               transactionRows=("quantity", "size"),
               productDescription=("ProductID", "first"))
    )

    product_meta = (
        df.sort_values("saleDate")
          .groupby("ProductID", as_index=False)
          .agg(firstSaleDate=("saleDate", "min"),
               lastSaleDate=("saleDate", "max"),
               totalUnitsSold=("quantity", "sum"),
               transactionRows=("quantity", "size"))
    )

    records = []
    for row in product_meta.itertuples(index=False):
        pid = row.ProductID
        p = daily.loc[daily["ProductID"] == pid, ["saleDate", "dailyUnits"]]
        # Create a calendar-day series only over this product's observed life.
        # Days before first observed sale are not assumed to be zero-demand days.
        idx = pd.date_range(row.firstSaleDate, max_date, freq="D")
        series = p.set_index("saleDate")["dailyUnits"].reindex(idx, fill_value=0.0).astype(float)

        recent = series.loc[series.index >= recent_start]
        long = series.loc[series.index >= long_start]
        recent30 = series.loc[series.index >= recent30_start]
        active_days_90 = int((long > 0).sum())
        active_days_28 = int((recent > 0).sum())

        # Weighted historical rate; recent demand has greater influence.
        recent_rate = float(recent.mean()) if len(recent) else 0.0
        long_rate = float(long.mean()) if len(long) else 0.0
        estimated_daily_demand = 0.60 * recent_rate + 0.40 * long_rate

        # Demand variability is computed on the long window; zero-sales days count
        # only after the product's first observed sale.
        demand_std = float(long.std(ddof=1)) if len(long) > 1 else 0.0
        if not np.isfinite(demand_std):
            demand_std = 0.0

        safety_stock = SERVICE_LEVEL_Z * demand_std * math.sqrt(LEAD_TIME_DAYS)
        reorder_point = estimated_daily_demand * LEAD_TIME_DAYS + safety_stock
        target_stock = estimated_daily_demand * (LEAD_TIME_DAYS + REVIEW_PERIOD_DAYS) + safety_stock
        expected_horizon_demand = estimated_daily_demand * PLANNING_HORIZON_DAYS

        days_since_last_sale = int((max_date - row.lastSaleDate).days)
        if days_since_last_sale <= 30 and active_days_28 > 0:
            status = "Active - review replenishment"
        elif days_since_last_sale <= 90 and active_days_90 > 0:
            status = "Intermittent / slowing"
        else:
            status = "No recent sales - review before ordering"

        records.append({
            "ProductID": pid,
            "firstSaleDate": row.firstSaleDate.date().isoformat(),
            "lastSaleDate": row.lastSaleDate.date().isoformat(),
            "daysSinceLastSale": days_since_last_sale,
            "transactionRows": int(row.transactionRows),
            "totalUnitsSoldHistorical": float(row.totalUnitsSold),
            "unitsSoldLast28Days": float(recent.sum()),
            "unitsSoldLast90Days": float(long.sum()),
            "activeSalesDaysLast28": active_days_28,
            "activeSalesDaysLast90": active_days_90,
            "averageDailyDemandLast28": recent_rate,
            "averageDailyDemandLast90": long_rate,
            "estimatedDailyDemand": estimated_daily_demand,
            "demandStdDevDailyLast90": demand_std,
            "estimatedDemandNext30Days": expected_horizon_demand,
            "assumedLeadTimeDays": LEAD_TIME_DAYS,
            "assumedReviewPeriodDays": REVIEW_PERIOD_DAYS,
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
    numeric_cols = result.select_dtypes(include=["number"]).columns
    result[numeric_cols] = result[numeric_cols].round(3)
    result = result.sort_values(
        ["recommendationStatus", "estimatedReplenishmentUnitsIfStockZero"],
        ascending=[True, False]
    )

    output_csv = REPORT_DIR / "inventoryOptimizationRecommendations.csv"
    result.to_csv(output_csv, index=False)

    # Summary statistics are descriptive; no real stockout/cost savings can be measured.
    active = result[result["recommendationStatus"] == "Active - review replenishment"]
    no_recent = result[result["recommendationStatus"] == "No recent sales - review before ordering"]
    summary = {
        "module": "RetailPulse inventory optimization baseline",
        "input": str(DATA_PATH.relative_to(ROOT)),
        "sourceDateRange": {
            "start": min_date.date().isoformat(),
            "end": max_date.date().isoformat()
        },
        "qualifyingTransactionRowsAfterCrossSheetCopyExclusion": int(len(df)),
        "uniqueProductsAnalyzed": int(result["ProductID"].nunique()),
        "productsWithSalesInLast28Days": int((result["activeSalesDaysLast28"] > 0).sum()),
        "productsWithSalesInLast90Days": int((result["activeSalesDaysLast90"] > 0).sum()),
        "productsFlaggedActiveReview": int(len(active)),
        "productsWithNoRecentSalesReview": int(len(no_recent)),
        "planningAssumptions": {
            "leadTimeDays": LEAD_TIME_DAYS,
            "reviewPeriodDays": REVIEW_PERIOD_DAYS,
            "planningHorizonDays": PLANNING_HORIZON_DAYS,
            "serviceLevelApprox": 0.95,
            "zScore": SERVICE_LEVEL_Z,
            "recentWindowDays": RECENT_WINDOW_DAYS,
            "longWindowDays": LONG_WINDOW_DAYS
        },
        "limitations": [
            "StockQuantity is missing, so actual on-hand inventory and order quantities cannot be determined.",
            "UnitCost is missing, so holding costs, order costs, and monetary savings cannot be calculated.",
            "Demand estimates use historical product-level sales rates, not product-level Prophet/LSTM forecasts.",
            "Lead time and review period are illustrative assumptions, not verified supplier parameters.",
            "The source ends on 2011-12-09 and is not live inventory data.",
            "No reduction in stockouts, overstock, or costs is claimed."
        ],
        "outputCsv": str(output_csv.relative_to(ROOT))
    }

    report_json = REPORT_DIR / "inventoryOptimizationSummary.json"
    report_txt = REPORT_DIR / "inventoryOptimizationReport.txt"
    report_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "RETAILPULSE INVENTORY OPTIMIZATION BASELINE",
        "=" * 46,
        f"Source date range: {summary['sourceDateRange']['start']} to {summary['sourceDateRange']['end']}",
        f"Qualifying transaction rows after excluding flagged cross-sheet copies: {len(df):,}",
        f"Unique products analyzed: {result['ProductID'].nunique():,}",
        f"Products with sales in last 28 days: {summary['productsWithSalesInLast28Days']:,}",
        f"Products with sales in last 90 days: {summary['productsWithSalesInLast90Days']:,}",
        f"Products flagged for active replenishment review: {len(active):,}",
        f"Products with no recent sales (review before ordering): {len(no_recent):,}",
        "",
        "ASSUMPTIONS",
        f"- Lead time: {LEAD_TIME_DAYS} days",
        f"- Review period: {REVIEW_PERIOD_DAYS} days",
        f"- Planning horizon: {PLANNING_HORIZON_DAYS} days",
        "- Service level: approximately 95% (normal-approximation z-score 1.645)",
        "- Demand rate: 60% recent 28-day average + 40% 90-day average",
        "",
        "INTERPRETATION",
        "- Reorder point = estimated daily demand × assumed lead time + estimated safety stock.",
        "- Target stock covers assumed lead time plus review period, plus estimated safety stock.",
        "- 'Replenishment units if stock zero' is a scenario, NOT an actual purchase order.",
        "- Product-level rates are historical estimates, not Prophet/LSTM product forecasts.",
        "",
        "LIMITATIONS",
    ]
    lines.extend([f"- {item}" for item in summary["limitations"]])
    lines += ["", f"Recommendations CSV: {output_csv.relative_to(ROOT)}"]
    report_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Figure 1: top products by estimated 30-day demand.
    top = result.sort_values("estimatedDemandNext30Days", ascending=False).head(15).sort_values(
        "estimatedDemandNext30Days", ascending=True
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["ProductID"], top["estimatedDemandNext30Days"])
    ax.set_title("Top 15 Products by Estimated Demand Over Next 30 Days")
    ax.set_xlabel("Estimated units (historical-rate baseline)")
    ax.set_ylabel("Product ID")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "inventoryTop30DayDemand.png", dpi=160)
    plt.close(fig)

    # Figure 2: products by recency category.
    counts = result["recommendationStatus"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(counts.index, counts.values)
    ax.set_title("Product Review Categories")
    ax.set_ylabel("Number of products")
    ax.tick_params(axis="x", labelrotation=15)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "inventoryProductReviewCategories.png", dpi=160)
    plt.close(fig)

    print("Inventory optimization baseline completed.")
    print(f"Date range: {min_date.date()} to {max_date.date()}")
    print(f"Qualifying transaction rows: {len(df):,}")
    print(f"Unique products analyzed: {result['ProductID'].nunique():,}")
    print(f"Products with sales in last 28 days: {summary['productsWithSalesInLast28Days']:,}")
    print(f"Recommendations: {output_csv}")
    print(f"Report: {report_txt}")
    print(f"Summary JSON: {report_json}")
    print(f"Figures: {FIGURE_DIR / 'inventoryTop30DayDemand.png'}")
    print(f"         {FIGURE_DIR / 'inventoryProductReviewCategories.png'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
