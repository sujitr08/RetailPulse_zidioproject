from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FILE = ROOT / "data" / "processedData" / "cleanedTransactions.csv"

df = pd.read_csv(FILE, low_memory=False)
df["invoiceDate"] = pd.to_datetime(df["invoiceDate"], errors="coerce")

signatureColumns = [
    "invoice", "stockCode", "description", "quantity",
    "invoiceDate", "unitPrice", "customerId", "country"
]

print("=== DATE RANGE BY SOURCE SHEET ===")
for sheet, group in df.groupby("sourceSheet", dropna=False):
    print(
        f"{sheet}: {len(group):,} rows | "
        f"{group['invoiceDate'].min()} to {group['invoiceDate'].max()}"
    )

print("\n=== ROW COUNTS DURING OVERLAP ===")
start = pd.Timestamp("2010-12-01")
end = pd.Timestamp("2010-12-09 23:59:59")
overlap = df[df["invoiceDate"].between(start, end)]

for sheet, group in overlap.groupby("sourceSheet", dropna=False):
    print(f"{sheet}: {len(group):,} rows")

print("\n=== EXACT CROSS-SHEET MATCHES ===")
left = df[df["sourceSheet"] == "Year 2009-2010"]
right = df[df["sourceSheet"] == "Year 2010-2011"]

# Count identical signatures separately in each sheet.
leftCounts = (
    left.groupby(signatureColumns, dropna=False)
    .size().reset_index(name="leftCount")
)
rightCounts = (
    right.groupby(signatureColumns, dropna=False)
    .size().reset_index(name="rightCount")
)

matches = leftCounts.merge(
    rightCounts,
    on=signatureColumns,
    how="inner"
)

matches["matchedPairs"] = matches[
    ["leftCount", "rightCount"]
].min(axis=1)

print(f"Matching signatures: {len(matches):,}")
print(f"Matching row pairs: {int(matches['matchedPairs'].sum()):,}")
print(
    "Excess repeated occurrences across matched signatures:",
    int(
        (matches["leftCount"] + matches["rightCount"] - 1).sum()
    )
)
print("\n=== SAMPLE MATCHES ===")
print(matches.head(10).to_string(index=False))
print("\nNo files were modified.")