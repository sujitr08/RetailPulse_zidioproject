from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FILE = ROOT / "data" / "processedData" / "cleanedTransactions.csv"

df = pd.read_csv(FILE, low_memory=False)

signatureColumns = [
    "invoice", "stockCode", "description", "quantity",
    "invoiceDate", "unitPrice", "customerId", "country"
]

missing = [c for c in signatureColumns + ["sourceSheet"] if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

print("=== DUPLICATE INVESTIGATION ===")
print(f"Total rows: {len(df):,}")

withinSheetDuplicates = 0
for sheet, group in df.groupby("sourceSheet", dropna=False):
    count = int(group.duplicated(
        subset=signatureColumns, keep="first"
    ).sum())
    withinSheetDuplicates += count
    print(f"Within {sheet}: {count:,}")

combinedDuplicates = int(
    df.duplicated(subset=signatureColumns, keep="first").sum()
)
print(f"Within-sheet duplicates combined: {withinSheetDuplicates:,}")
print(f"Combined duplicates: {combinedDuplicates:,}")
print(
    "Difference to investigate:",
    f"{combinedDuplicates - withinSheetDuplicates:,}"
)

# Identify transaction signatures appearing in both source sheets.
# dropna=False ensures missing values are included in grouping.
signatureCounts = (
    df.groupby(signatureColumns + ["sourceSheet"], dropna=False)
      .size()
      .rename("rows")
      .reset_index()
)

sheetCounts = (
    signatureCounts.groupby(signatureColumns, dropna=False)
    .agg(
        sourceSheetCount=("sourceSheet", "nunique"),
        totalOccurrences=("rows", "sum"),
        sheetOccurrences=("rows", "size"),
    )
    .reset_index()
)

crossSheet = sheetCounts[sheetCounts["sourceSheetCount"] > 1].copy()

print("\n=== CROSS-SHEET OVERLAP ===")
print(f"Signatures appearing in multiple sheets: {len(crossSheet):,}")
print(
    "Rows belonging to those signatures:",
    f"{int(crossSheet['totalOccurrences'].sum()):,}"
)
print(
    "Excess occurrences for cross-sheet signatures:",
    f"{int((crossSheet['totalOccurrences'] - 1).sum()):,}"
)

print("\nSample cross-sheet duplicate signatures:")
print(crossSheet.head(15).to_string(index=False))

print("\nNo files were modified.")
