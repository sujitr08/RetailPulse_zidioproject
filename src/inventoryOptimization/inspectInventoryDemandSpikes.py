from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "processedData" / "cleanedTransactions.csv"
OUT_DIR = ROOT / "reports" / "modelReports"
OUT_TXT = OUT_DIR / "inventoryDemandSpikeInvestigation.txt"
OUT_CSV = OUT_DIR / "inventoryDemandSpikeDailyDetails.csv"

PRODUCTS_TO_INSPECT = [
    "PROD02919", "PROD01746", "PROD02825", "PROD00630",
    "PROD03822", "PROD01240", "PROD01458",
]

def find_col(df, options, required=True):
    normalized = {str(c).strip().lower().replace("_", "").replace(" ", ""): c for c in df.columns}
    for option in options:
        key = option.lower().replace("_", "").replace(" ", "")
        if key in normalized:
            return normalized[key]
    if required:
        raise ValueError(f"Missing expected column among {options}. Found: {list(df.columns)}")
    return None

def true_mask(series):
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])

def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Could not find {SOURCE}")

    df = pd.read_csv(SOURCE, low_memory=False)
    product_col = find_col(df, ["ProductID", "product_id"])
    date_col = find_col(df, ["InvoiceDate", "invoice_date"])
    qty_col = find_col(df, ["Quantity", "quantity"])
    invoice_col = find_col(df, ["Invoice", "InvoiceNo", "invoice_no"], required=False)
    price_col = find_col(df, ["Price", "UnitPrice", "unit_price"], required=False)
    description_col = find_col(df, ["Description", "description"], required=False)
    candidate_col = find_col(df, ["IsCandidateSale", "candidateSale", "is_candidate_sale"], required=False)
    cross_col = find_col(df, ["IsCrossSheetDuplicate", "crossSheetDuplicate", "is_cross_sheet_duplicate"], required=False)

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[qty_col] = pd.to_numeric(df[qty_col], errors="coerce")
    df = df[df[product_col].astype(str).isin(PRODUCTS_TO_INSPECT)].copy()
    df = df.dropna(subset=[date_col, qty_col])
    df["_day"] = df[date_col].dt.normalize()

    # Match the optimizer's qualifying-sale policy when its flags are present.
    if candidate_col:
        df = df[true_mask(df[candidate_col])]
    else:
        df = df[df[qty_col] > 0]
    if cross_col:
        df = df[~true_mask(df[cross_col])]

    if df.empty:
        raise ValueError("No qualifying transaction rows found for the requested products.")

    end_date = df["_day"].max()
    start_28 = end_date - pd.Timedelta(days=27)
    recent = df[(df["_day"] >= start_28) & (df["_day"] <= end_date)].copy()

    daily = recent.groupby([product_col, "_day"], as_index=False).agg(
        unitsSold=(qty_col, "sum"),
        transactionRows=(qty_col, "size"),
        distinctInvoices=(invoice_col, "nunique") if invoice_col else (qty_col, "size"),
    )
    daily["peakDayShareWithinProduct28d"] = daily["unitsSold"] / daily.groupby(product_col)["unitsSold"].transform("sum")
    daily = daily.sort_values([product_col, "unitsSold"], ascending=[True, False])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUT_CSV, index=False)

    lines = [
        "RETAILPULSE INVENTORY DEMAND SPIKE INVESTIGATION",
        "=" * 48,
        f"Source: {SOURCE.relative_to(ROOT)}",
        f"Qualifying source rows for selected products: {len(df)}",
        f"Recent 28-day window: {start_28.date()} to {end_date.date()}",
        "Cross-sheet duplicate copies excluded when the flag is available.",
        "",
        "DAILY SALES DETAIL (largest days per product)",
    ]
    for product in PRODUCTS_TO_INSPECT:
        product_daily = daily[daily[product_col].astype(str) == product].head(5)
        lines.append("")
        lines.append(f"{product}:")
        if product_daily.empty:
            lines.append("  No qualifying sales in the last 28 days.")
            continue
        for _, row in product_daily.iterrows():
            lines.append(
                f"  {row['_day'].date()} | units={row['unitsSold']:.3f} | "
                f"rows={int(row['transactionRows'])} | invoices={int(row['distinctInvoices'])} | "
                f"shareOf28d={row['peakDayShareWithinProduct28d']:.3f}"
            )

    lines.extend(["", "TOP-DAY TRANSACTION LINES FOR THE HIGHEST-SPIKE PRODUCT", ""])
    target = "PROD02919"
    target_daily = daily[daily[product_col].astype(str) == target]
    if target_daily.empty:
        lines.append(f"No qualifying recent transactions found for {target}.")
    else:
        peak_day = target_daily.sort_values("unitsSold", ascending=False).iloc[0]["_day"]
        detail = recent[
            (recent[product_col].astype(str) == target) & (recent["_day"] == peak_day)
        ].copy()
        show_cols = []
        for col in [date_col, invoice_col, product_col, description_col, qty_col, price_col]:
            if col and col in detail.columns and col not in show_cols:
                show_cols.append(col)
        lines.append(f"Peak day: {peak_day.date()}; qualifying rows on that day: {len(detail)}")
        lines.append(f"Units summed on peak day: {detail[qty_col].sum():.3f}")
        if show_cols:
            for _, row in detail[show_cols].head(40).iterrows():
                parts = []
                for col in show_cols:
                    value = row[col]
                    parts.append(f"{col}={value}")
                lines.append("  " + " | ".join(parts))
            if len(detail) > 40:
                lines.append(f"  ... {len(detail) - 40} additional rows omitted; inspect the CSV/source for full details.")

    lines.extend([
        "",
        "INTERPRETATION NOTES",
        "- Large daily totals may reflect a genuine bulk order; verify invoice, quantity, and product description.",
        "- Do not automatically cap, remove, or convert flagged demand without source-level evidence.",
        "- This investigation uses historical sales only; it cannot establish current inventory or future supplier availability.",
        "",
        f"Daily detail CSV: {OUT_CSV.relative_to(ROOT)}",
    ])
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Investigation completed. Window end date: {end_date.date()}")
    print(f"Report: {OUT_TXT}")
    print(f"Daily details CSV: {OUT_CSV}")

if __name__ == "__main__":
    main()
