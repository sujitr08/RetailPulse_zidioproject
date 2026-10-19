from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "processedData" / "cleanedTransactions.csv"

df = pd.read_csv(
    DATA_FILE,
    low_memory=False,
    parse_dates=["invoiceDate"]
)

print("\n=== 1. DATASET OVERVIEW ===")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Distinct invoices: {df['invoice'].nunique():,}")
print(f"Distinct products: {df['stockCode'].nunique():,}")
print(f"Distinct identified customers: {df['customerId'].nunique():,}")

print("\n=== 2. DUPLICATE FLAGS ===")
if "isExactDuplicate" in df.columns:
    print(df["isExactDuplicate"].value_counts(dropna=False))
    print(
        "Flagged duplicate rows:",
        int(df["isExactDuplicate"].astype(str).str.lower()
            .isin(["true", "1"]).sum())
    )

duplicateColumns = [
    "invoice", "stockCode", "description", "quantity",
    "invoiceDate", "unitPrice", "customerId", "country"
]
availableColumns = [c for c in duplicateColumns if c in df.columns]

print(
    "Duplicates across normalized transaction columns:",
    int(df.duplicated(subset=availableColumns, keep="first").sum())
)

if "sourceSheet" in df.columns:
    print("\nDuplicates within each source sheet:")
    for sheet, group in df.groupby("sourceSheet", dropna=False):
        count = group.duplicated(
            subset=availableColumns, keep="first"
        ).sum()
        print(f"{sheet}: {count:,}")

print("\n=== 3. DATA-QUALITY FLAGS ===")
flagColumns = [
    "hasMissingCustomerId",
    "hasMissingDescription",
    "hasInvalidDate",
    "hasNonPositiveQuantity",
    "hasNonPositivePrice",
    "isCancellationInvoice",
    "isCandidateSale",
]

for column in flagColumns:
    if column in df.columns:
        values = (
            df[column].astype(str).str.strip().str.lower()
            .isin(["true", "1"])
        )
        print(f"{column}: {int(values.sum()):,}")

print("\n=== 4. TRANSACTION AMOUNT DISTRIBUTION ===")
print(df["SalesAmount"].describe(
    percentiles=[0.01, 0.05, 0.50, 0.95, 0.99, 0.999]
).to_string())

print("\nLargest positive transaction amounts:")
columns = [
    "invoice", "stockCode", "description", "quantity",
    "unitPrice", "SalesAmount", "customerId", "invoiceDate"
]
print(
    df.loc[df["SalesAmount"] > 0]
      .nlargest(15, "SalesAmount")[columns]
      .to_string(index=False)
)

print("\n=== 5. NEGATIVE AND ZERO TRANSACTIONS ===")
for label, condition in [
    ("Negative quantity", df["quantity"] < 0),
    ("Zero quantity", df["quantity"] == 0),
    ("Negative unit price", df["unitPrice"] < 0),
    ("Zero unit price", df["unitPrice"] == 0),
    ("Negative sales amount", df["SalesAmount"] < 0),
    ("Zero sales amount", df["SalesAmount"] == 0),
]:
    print(f"{label}: {int(condition.sum()):,}")

print("\nNegative sales amount examples:")
print(
    df.loc[df["SalesAmount"] < 0, columns]
      .head(15)
      .to_string(index=False)
)

print("\n=== 6. MISSING VALUES IN IMPORTANT FIELDS ===")
for column in [
    "invoice", "stockCode", "description", "quantity",
    "invoiceDate", "unitPrice", "customerId", "country"
]:
    if column in df.columns:
        print(f"{column}: {int(df[column].isna().sum()):,}")

print("\nDiagnostic completed. No source data was modified.")
