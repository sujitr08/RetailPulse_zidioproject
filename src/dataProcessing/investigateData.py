from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "rawData" / "onlineRetail.xlsx"
REPORT_DIR = ROOT / "reports" / "modelReports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

output = []

def show(title, value):
    output.append(f"\n{'=' * 65}\n{title}\n{'=' * 65}")
    output.append(str(value))

sheets = pd.read_excel(DATA_FILE, sheet_name=None, engine="openpyxl")
df = pd.concat(sheets.values(), ignore_index=True)

show("COMBINED DATASET", f"Rows: {len(df):,}\nColumns: {df.columns.tolist()}")

dates = pd.to_datetime(df["InvoiceDate"], errors="coerce")
show(
    "MONTHLY DATE COVERAGE",
    dates.dt.to_period("M").value_counts().sort_index().to_string()
)

invoice_text = df["Invoice"].astype(str).str.strip()
show(
    "INVOICE IDENTIFIERS",
    "Invoices beginning with C: "
    f"{invoice_text.str.upper().str.startswith('C').sum():,}\n"
    "Missing invoice IDs: "
    f"{df['Invoice'].isna().sum():,}"
)

show(
    "EXACT DUPLICATE EXAMPLES",
    df[df.duplicated(keep=False)]
    .sort_values(df.columns.tolist())
    .head(20)
    .to_string(index=False)
)

show(
    "NON-POSITIVE QUANTITY EXAMPLES",
    df.loc[df["Quantity"] <= 0, [
        "Invoice", "StockCode", "Description", "Quantity",
        "InvoiceDate", "Price", "Customer ID"
    ]].head(20).to_string(index=False)
)

show(
    "ZERO OR NEGATIVE PRICE EXAMPLES",
    df.loc[df["Price"] <= 0, [
        "Invoice", "StockCode", "Description", "Quantity",
        "InvoiceDate", "Price", "Customer ID"
    ]].head(30).to_string(index=False)
)

show(
    "MISSING CUSTOMER ID — QUANTITY AND PRICE SUMMARY",
    df.loc[df["Customer ID"].isna(), ["Quantity", "Price"]]
    .describe()
    .to_string()
)

show(
    "MISSING DESCRIPTION — STOCK CODE COUNTS",
    df.loc[df["Description"].isna(), "StockCode"]
    .value_counts()
    .head(20)
    .to_string()
)

report_file = REPORT_DIR / "dataInvestigation.txt"
report_file.write_text("\n".join(output), encoding="utf-8")

print("\n".join(output))
print(f"\nInvestigation saved to: {report_file}")
