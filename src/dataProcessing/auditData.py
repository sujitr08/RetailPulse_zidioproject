from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "rawData" / "onlineRetail.xlsx"
REPORT_DIR = ROOT / "reports" / "modelReports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_COLUMNS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
]

report = []

def record(message=""):
    report.append(str(message))

record("RETAILPULSE — RAW DATA AUDIT")
record("=" * 60)
record(f"Source file: {DATA_FILE}")
record(f"File exists: {DATA_FILE.exists()}")

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

with pd.ExcelFile(DATA_FILE, engine="openpyxl") as workbook:
    record(f"Worksheets: {workbook.sheet_names}")

    for sheet_name in workbook.sheet_names:
        record("\n" + "=" * 60)
        record(f"SHEET: {sheet_name}")

        df = pd.read_excel(
            workbook,
            sheet_name=sheet_name,
            engine="openpyxl",
        )

        record(f"Shape: {df.shape}")
        record(f"Columns: {df.columns.tolist()}")
        record(f"Data types:\n{df.dtypes.to_string()}")

        missing_columns = [
            column for column in EXPECTED_COLUMNS
            if column not in df.columns
        ]

        if missing_columns:
            record(f"ERROR — Missing expected columns: {missing_columns}")
            continue

        record("\nMissing values:")
        record(df.isna().sum().to_string())

        record(f"\nExact duplicate rows: {df.duplicated().sum()}")

        record("\nNumeric summary:")
        record(
            df[["Quantity", "Price"]]
            .describe()
            .to_string()
        )

        record(f"\nQuantity <= 0: {(df['Quantity'] <= 0).sum()}")
        record(f"Price < 0: {(df['Price'] < 0).sum()}")
        record(f"Price == 0: {(df['Price'] == 0).sum()}")

        dates = pd.to_datetime(df["InvoiceDate"], errors="coerce")
        record(f"\nUnparseable dates: {dates.isna().sum()}")
        record(f"Earliest date: {dates.min()}")
        record(f"Latest date: {dates.max()}")

        record(
            "\nUnique invoices: "
            f"{df['Invoice'].nunique(dropna=True)}"
        )
        record(
            "Unique stock codes: "
            f"{df['StockCode'].nunique(dropna=True)}"
        )
        record(
            "Unique customers: "
            f"{df['Customer ID'].nunique(dropna=True)}"
        )
        record(
            "Unique countries: "
            f"{df['Country'].nunique(dropna=True)}"
        )

        record("\nFirst five records:")
        record(df.head().to_string(index=False))

        del df

output_file = REPORT_DIR / "dataAudit.txt"
output_file.write_text("\n".join(report), encoding="utf-8")

print("\n".join(report))
print(f"\nFull audit saved to: {output_file}")
