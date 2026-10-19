from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "processedData" / "cleanedTransactions.csv"
REPORT_DIR = ROOT / "reports" / "modelReports"
OUTPUT_CSV = REPORT_DIR / "inventoryDemandConfidenceAudit.csv"
OUTPUT_TXT = REPORT_DIR / "inventoryDemandConfidenceAudit.txt"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("retailPulse.inventoryConfidence")


def resolve_column(frame: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    lookup = {str(col).strip().lower().replace("_", "").replace(" ", ""): col for col in frame.columns}
    for candidate in candidates:
        key = candidate.strip().lower().replace("_", "").replace(" ", "")
        if key in lookup:
            return lookup[key]
    if required:
        raise ValueError(
            f"Could not find any of {candidates}. Available columns: {list(frame.columns)}"
        )
    return None


def main() -> int:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing cleaned transaction file: {DATA_FILE}\n"
            "Run the existing cleaning pipeline first."
        )

    logger.info("Reading cleaned transactions: %s", DATA_FILE)
    frame = pd.read_csv(DATA_FILE, low_memory=False)

    product_col = resolve_column(frame, ["ProductID", "product_id", "productId"])
    date_col = resolve_column(frame, ["InvoiceDate", "invoice_date", "date"])
    quantity_col = resolve_column(frame, ["Quantity", "quantity"])
    candidate_col = resolve_column(
        frame, ["IsCandidateSale", "candidateSale", "is_candidate_sale"], required=False
    )
    cross_dup_col = resolve_column(
        frame, ["IsCrossSheetDuplicate", "crossSheetDuplicate", "is_cross_sheet_duplicate"],
        required=False,
    )
    within_dup_col = resolve_column(
        frame, ["IsWithinSheetDuplicate", "withinSheetDuplicate", "is_within_sheet_duplicate"],
        required=False,
    )
    cancellation_col = resolve_column(
        frame, ["IsCancellation", "isCancellation", "is_cancellation"], required=False
    )

    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame[quantity_col] = pd.to_numeric(frame[quantity_col], errors="coerce")
    frame = frame.dropna(subset=[product_col, date_col, quantity_col]).copy()

    # Reuse the existing sale definition when available. Otherwise apply a
    # conservative fallback without deleting or changing the source file.
    if candidate_col:
        candidate = frame[candidate_col].astype(str).str.strip().str.lower().isin(
            ["true", "1", "yes"]
        )
        frame = frame[candidate].copy()
    else:
        frame = frame[frame[quantity_col] > 0].copy()
        if cancellation_col:
            cancelled = frame[cancellation_col].astype(str).str.strip().str.lower().isin(
                ["true", "1", "yes"]
            )
            frame = frame[~cancelled].copy()

    # Avoid counting known cross-sheet copies twice, matching the existing
    # inventory baseline. Within-sheet duplicates are not silently removed.
    if cross_dup_col:
        cross_dup = frame[cross_dup_col].astype(str).str.strip().str.lower().isin(
            ["true", "1", "yes"]
        )
        frame = frame[~cross_dup].copy()

    if frame.empty:
        raise ValueError("No qualifying sale rows remain after applying sale and duplicate filters.")

    frame["_date"] = frame[date_col].dt.normalize()
    end_date = frame["_date"].max()
    start_28 = end_date - pd.Timedelta(days=27)
    start_90 = end_date - pd.Timedelta(days=89)

    recent_28 = frame[(frame["_date"] >= start_28) & (frame["_date"] <= end_date)]
    recent_90 = frame[(frame["_date"] >= start_90) & (frame["_date"] <= end_date)]

    # Sales are summed per product-day. Zero-sale calendar days are reflected
    # in the fixed 28- and 90-day denominators.
    daily_28 = (
        recent_28.groupby([product_col, "_date"], as_index=False)[quantity_col]
        .sum()
        .rename(columns={quantity_col: "unitsSold"})
    )
    summary_28 = daily_28.groupby(product_col).agg(
        unitsSoldLast28Days=("unitsSold", "sum"),
        activeSalesDaysLast28=("_date", "nunique"),
        peakDayUnitsLast28=("unitsSold", "max"),
    )
    summary_28["averageDailyDemandLast28"] = summary_28["unitsSoldLast28Days"] / 28.0
    summary_28["peakDayShareLast28"] = np.where(
        summary_28["unitsSoldLast28Days"] > 0,
        summary_28["peakDayUnitsLast28"] / summary_28["unitsSoldLast28Days"],
        np.nan,
    )

    summary_90 = (
        recent_90.groupby(product_col)[quantity_col]
        .sum()
        .rename("unitsSoldLast90Days")
        .to_frame()
    )
    summary_90["averageDailyDemandLast90"] = summary_90["unitsSoldLast90Days"] / 90.0

    # Preserve every product seen in the source, including products with no
    # qualifying recent sales, so the report does not hide dormant products.
    all_products = pd.Index(frame[product_col].dropna().unique(), name=product_col)
    result = pd.DataFrame(index=all_products)
    result = result.join(summary_28, how="left").join(summary_90, how="left")
    for col in [
        "unitsSoldLast28Days", "activeSalesDaysLast28", "peakDayUnitsLast28",
        "averageDailyDemandLast28", "peakDayShareLast28", "unitsSoldLast90Days",
        "averageDailyDemandLast90",
    ]:
        if col in result.columns:
            result[col] = result[col].fillna(0)
    result["estimatedDailyDemandBaseline"] = (
        0.60 * result["averageDailyDemandLast28"]
        + 0.40 * result["averageDailyDemandLast90"]
    )
    result["recentToLongerRateRatio"] = np.where(
        result["averageDailyDemandLast90"] > 0,
        result["averageDailyDemandLast28"] / result["averageDailyDemandLast90"],
        np.where(result["averageDailyDemandLast28"] > 0, np.inf, 0.0),
    )

    # Transparent review rules, not automatic deletion or proof of bad data:
    # (1) one or two active days with most recent-window sales on one day;
    # (2) recent daily rate >3x the 90-day daily rate with <=3 active days;
    # (3) high concentration of recent sales in one day.
    low_activity_spike = (
        (result["activeSalesDaysLast28"] <= 2)
        & (result["unitsSoldLast28Days"] > 0)
        & (
            (result["peakDayShareLast28"] >= 0.80)
            | (result["recentToLongerRateRatio"] >= 3.0)
        )
    )
    concentrated_sales = (
        (result["peakDayShareLast28"] >= 0.80)
        & (result["unitsSoldLast28Days"] > 0)
    )
    result["manualReviewFlag"] = low_activity_spike | concentrated_sales
    result["demandConfidence"] = np.select(
        [
            result["manualReviewFlag"],
            (result["activeSalesDaysLast28"] >= 10)
            & (result["recentToLongerRateRatio"].between(0.5, 2.0)),
            result["activeSalesDaysLast28"] > 0,
        ],
        ["LOW - SPIKE/CONCENTRATION REVIEW", "HIGHER - RECENT ACTIVITY", "MEDIUM/LOW - LIMITED ACTIVITY"],
        default="LOW - NO RECENT SALES",
    )
    result["estimatedDemandNext30DaysBaseline"] = result["estimatedDailyDemandBaseline"] * 30.0
    result["auditWindowEndDate"] = str(end_date.date())
    result = result.reset_index().sort_values(
        ["manualReviewFlag", "estimatedDemandNext30DaysBaseline"],
        ascending=[False, False],
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False)

    flagged = result[result["manualReviewFlag"]]
    top_flagged = flagged.head(15)
    lines = [
        "RETAILPULSE INVENTORY DEMAND CONFIDENCE AUDIT",
        "=" * 44,
        f"Source file: {DATA_FILE.relative_to(ROOT)}",
        f"Rows read: {len(pd.read_csv(DATA_FILE, usecols=[product_col], low_memory=False))}",
        f"Qualifying transaction rows after filters: {len(frame)}",
        f"Products assessed: {len(result)}",
        f"Window end date: {end_date.date()}",
        "Demand baseline: 60% x 28-calendar-day average + 40% x 90-calendar-day average.",
        "Zero-sale calendar days are included in the fixed window denominators.",
        "",
        "REVIEW RULES",
        "- Flag products with low recent active-day count and a concentrated sales day or recent rate >=3x the 90-day rate.",
        "- Flag products when >=80% of 28-day units occurred on a single day.",
        "- Flags are review prompts, not automatic evidence of erroneous transactions.",
        "",
        f"Products flagged for manual review: {len(flagged)}",
        "",
        "TOP FLAGGED PRODUCTS",
        "ProductID | Units28d | ActiveDays28d | PeakDayShare | AvgDaily28 | AvgDaily90 | BaselineDaily | Baseline30d | Review",
    ]
    for _, row in top_flagged.iterrows():
        lines.append(
            f"{row[product_col]} | {row['unitsSoldLast28Days']:.3f} | "
            f"{row['activeSalesDaysLast28']:.0f} | {row['peakDayShareLast28']:.3f} | "
            f"{row['averageDailyDemandLast28']:.3f} | {row['averageDailyDemandLast90']:.3f} | "
            f"{row['estimatedDailyDemandBaseline']:.3f} | "
            f"{row['estimatedDemandNext30DaysBaseline']:.3f} | {row['demandConfidence']}"
        )
    lines.extend([
        "",
        "LIMITATIONS",
        "- This report does not overwrite existing replenishment recommendations.",
        "- Historical transaction rates are not product-level Prophet/LSTM forecasts.",
        "- The source data has no reliable stock-on-hand or unit-cost values.",
        "- Lead time and service level must be supplied and validated separately.",
        "- Review flags do not automatically remove or cap demand; investigate source transactions first.",
        "",
        f"CSV output: {OUTPUT_CSV.relative_to(ROOT)}",
    ])
    OUTPUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    logger.info("Products assessed: %s", len(result))
    logger.info("Products flagged for manual review: %s", len(flagged))
    logger.info("Audit CSV: %s", OUTPUT_CSV)
    logger.info("Audit report: %s", OUTPUT_TXT)
    logger.info("Window end date: %s", end_date.date())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
