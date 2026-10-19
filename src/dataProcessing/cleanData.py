"""
New field	Implementation
ProductID	: Stable ID mapped from StockCode
ProductCategory	: Transparent rule-based category, with Uncategorized when the rules cannot confidently classify a product
UnitCost	: Missing until reliable cost data is available; never presented as actual cost without evidence
StockQuantity	: Missing until genuine stock data is available; simulated stock will be kept separate if we need it for a demonstration
SalesAmount	: Quantity * Price, preserving returns and cancellationsv
"""
"""retailPulse data cleaning and feature engineering.
Creates normalized transactions, stable product IDs, heuristic categories,
data-quality flags, duplicate diagnostics, and a reproducible report.
No transaction rows are deleted. UnitCost and StockQuantity remain missing
because verified source values are not available in the source workbook.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = ROOT / "data" / "rawData" / "onlineRetail.xlsx"
PROCESSED_DIR = ROOT / "data" / "processedData"
REPORT_DIR = ROOT / "reports" / "modelReports"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = PROCESSED_DIR / "cleanedTransactions.csv"
PRODUCT_MAP_FILE = PROCESSED_DIR / "productIdMap.csv"
REPORT_FILE = REPORT_DIR / "cleaningReport.txt"

SOURCE_COLUMNS = [
    "Invoice", "StockCode", "Description", "Quantity", "InvoiceDate",
    "Price", "Customer ID", "Country",
]
COLUMN_MAP = {
    "Invoice": "invoice", "StockCode": "stockCode",
    "Description": "description", "Quantity": "quantity",
    "InvoiceDate": "invoiceDate", "Price": "unitPrice",
    "Customer ID": "customerId", "Country": "country",
}
NEWER_SHEET = "Year 2010-2011"
OLDER_SHEET = "Year 2009-2010"


def categorizeProduct(description):
    """Assign a transparent keyword-based category; uncertain items stay Uncategorized."""
    if pd.isna(description) or not str(description).strip():
        return "Uncategorized"
    text = str(description).upper()
    categoryRules = [
        ("Lighting", ["LIGHT", "LANTERN", "LAMP", "CANDLE", "T-LIGHT"]),
        ("HomeDecor", ["HOLDER", "FRAME", "VASE", "ORNAMENT", "GARLAND", "WALL", "DECORATION", "DOORMAT", "DOOR MAT"]),
        ("KitchenAndDining", ["MUG", "CUP", "PLATE", "BOWL", "CUTLERY", "NAPKIN", "TEA", "KITCHEN", "TRINKET BOX"]),
        ("ToysAndGames", ["TOY", "GAME", "JIGSAW", "PUZZLE", "PLAYING CARDS", "CHESS", "DOMINO"]),
        ("Accessories", ["BRACELET", "NECKLACE", "RING", "EARRING", "HAIRCLIP", "HAIR SLIDE", "HAIR COMB"]),
        ("StationeryAndGifts", ["CARD", "PAPER", "PENCIL", "NOTEBOOK", "GIFT", "RIBBON", "WRAP", "WRAPPING", "STICKER"]),
        ("Textiles", ["CUSHION", "BLANKET", "TOWEL", "FABRIC", "COTTON", "WOOL", "APRON", "BUNTING"]),
        ("GardenAndOutdoor", ["GARDEN", "PLANTER", "FLOWER", "BIRD", "OUTDOOR", "POTTING"]),
    ]
    for category, keywords in categoryRules:
        if any(keyword in text for keyword in keywords):
            return category
    return "Uncategorized"


# Load and validate all worksheets.
if not RAW_FILE.exists():
    raise FileNotFoundError(f"Raw dataset not found: {RAW_FILE}")
sheets = pd.read_excel(RAW_FILE, sheet_name=None, engine="openpyxl")
frames = []
for sheetName, sheetData in sheets.items():
    missingColumns = set(SOURCE_COLUMNS) - set(sheetData.columns)
    if missingColumns:
        raise ValueError(f"Worksheet '{sheetName}' is missing columns: {sorted(missingColumns)}")
    sheetData = sheetData.copy()
    sheetData["sourceSheet"] = sheetName
    frames.append(sheetData)
if not frames:
    raise ValueError("The workbook contains no readable worksheets.")
df = pd.concat(frames, ignore_index=True).rename(columns=COLUMN_MAP)
inputRows = len(df)
df["sourceSheet"] = df["sourceSheet"].astype("string").str.strip()

# Standardize types without discarding rows.
for column in ["invoice", "stockCode", "description", "country"]:
    df[column] = df[column].astype("string").str.strip()
df["description"] = df["description"].replace("", pd.NA)
df["customerId"] = pd.to_numeric(df["customerId"], errors="coerce").astype("Int64")
df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
df["unitPrice"] = pd.to_numeric(df["unitPrice"], errors="coerce")
df["invoiceDate"] = pd.to_datetime(df["invoiceDate"], errors="coerce")
for column, label in [("invoice", "Invoice"), ("stockCode", "StockCode")]:
    if df[column].isna().any():
        raise ValueError(f"{label} contains missing identifiers.")
for column, label in [("quantity", "Quantity"), ("unitPrice", "Price")]:
    if df[column].isna().any():
        raise ValueError(f"{label} contains invalid or missing values.")

# Stable product IDs.
stockCodes = sorted(df["stockCode"].dropna().unique().tolist())
productIdMap = pd.DataFrame({
    "stockCode": stockCodes,
    "ProductID": [f"PROD{number:05d}" for number in range(1, len(stockCodes) + 1)],
})
mapping = dict(zip(productIdMap["stockCode"], productIdMap["ProductID"]))
df["ProductID"] = df["stockCode"].map(mapping)
if df["ProductID"].isna().any():
    raise ValueError("Some stock codes did not receive a ProductID.")
productIdMap.to_csv(PRODUCT_MAP_FILE, index=False)

# Business fields; costs and inventory are unavailable in the source.
df["ProductCategory"] = df["description"].map(categorizeProduct)
df["UnitCost"] = pd.Series(pd.NA, index=df.index, dtype="Float64")
df["StockQuantity"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
df["SalesAmount"] = df["quantity"] * df["unitPrice"]

# Data-quality and modelling flags.
df["hasMissingCustomerId"] = df["customerId"].isna()
df["hasMissingDescription"] = df["description"].isna()
df["hasInvalidDate"] = df["invoiceDate"].isna()
df["hasNonPositiveQuantity"] = df["quantity"] <= 0
df["hasNonPositivePrice"] = df["unitPrice"] <= 0
invoiceText = df["invoice"].fillna("").astype("string").str.strip()
df["isCancellationInvoice"] = invoiceText.str.upper().str.startswith("C", na=False)
df["isCandidateSale"] = (
    (df["quantity"] > 0) & (df["unitPrice"] > 0)
    & df["invoiceDate"].notna() & (~df["isCancellationInvoice"])
)

# Duplicate signatures use the eight normalized transaction fields.
duplicateColumns = list(COLUMN_MAP.values())
df["isWithinSheetDuplicate"] = df.duplicated(
    subset=duplicateColumns + ["sourceSheet"], keep="first"
)

# Pair repeated signatures one-for-one across sheets using a stable hash and
# occurrence number within each sheet. This handles repeated signatures and NA.
df["_signatureHash"] = pd.util.hash_pandas_object(
    df[duplicateColumns], index=False
).astype("uint64")
df["_occurrenceWithinSheet"] = df.groupby(
    ["sourceSheet", "_signatureHash"], dropna=False, sort=False
).cumcount()
newerMask = df["sourceSheet"] == NEWER_SHEET
olderMask = df["sourceSheet"] == OLDER_SHEET
newerKeys = set(zip(
    df.loc[newerMask, "_signatureHash"].tolist(),
    df.loc[newerMask, "_occurrenceWithinSheet"].tolist(),
))
olderKeys = zip(
    df.loc[olderMask, "_signatureHash"].tolist(),
    df.loc[olderMask, "_occurrenceWithinSheet"].tolist(),
)
olderCrossDuplicate = pd.Series(
    [key in newerKeys for key in olderKeys], index=df.index[olderMask], dtype=bool
)
df["isCrossSheetDuplicate"] = False
df.loc[olderCrossDuplicate.index, "isCrossSheetDuplicate"] = olderCrossDuplicate
df["isExactDuplicate"] = df["isWithinSheetDuplicate"] | df["isCrossSheetDuplicate"]
df.drop(columns=["_signatureHash", "_occurrenceWithinSheet"], inplace=True)

# Validate and save. This script flags duplicates but deliberately removes no rows.
requiredFields = [
    "ProductID", "ProductCategory", "UnitCost", "StockQuantity", "SalesAmount",
    "hasMissingCustomerId", "hasMissingDescription", "hasInvalidDate",
    "hasNonPositiveQuantity", "hasNonPositivePrice", "isCancellationInvoice",
    "isCandidateSale", "isWithinSheetDuplicate", "isCrossSheetDuplicate", "isExactDuplicate",
]
missingFields = set(requiredFields) - set(df.columns)
if missingFields:
    raise ValueError(f"Required fields missing: {sorted(missingFields)}")
if len(df) != inputRows:
    raise ValueError(f"Row count changed unexpectedly: {inputRows} -> {len(df)}")
df.to_csv(OUTPUT_FILE, index=False)
savedHeader = pd.read_csv(OUTPUT_FILE, nrows=0).columns.tolist()
if savedHeader != df.columns.tolist():
    raise ValueError("Saved CSV schema does not match the in-memory data.")

withinCount = int(df["isWithinSheetDuplicate"].sum())
crossCount = int(df["isCrossSheetDuplicate"].sum())
anyDuplicateCount = int(df["isExactDuplicate"].sum())
reportLines = [
    "RETAILPULSE - CLEANING AND FEATURE REPORT", "=" * 60,
    f"Input rows: {inputRows:,}", f"Output rows: {len(df):,}",
    f"Rows removed: {inputRows - len(df):,}", f"Output columns: {len(df.columns)}",
    f"Distinct products: {df['stockCode'].nunique():,}",
    f"Unique ProductIDs: {df['ProductID'].nunique():,}",
    f"Missing customer IDs: {int(df['hasMissingCustomerId'].sum()):,}",
    f"Missing descriptions: {int(df['hasMissingDescription'].sum()):,}",
    f"Invalid dates: {int(df['hasInvalidDate'].sum()):,}",
    f"Within-sheet duplicate rows flagged: {withinCount:,}",
    f"Cross-sheet duplicate rows flagged (older-sheet copies): {crossCount:,}",
    f"Rows flagged by either duplicate rule: {anyDuplicateCount:,}",
    f"Non-positive quantities: {int(df['hasNonPositiveQuantity'].sum()):,}",
    f"Non-positive prices: {int(df['hasNonPositivePrice'].sum()):,}",
    f"Cancellation invoice rows: {int(df['isCancellationInvoice'].sum()):,}",
    f"Candidate sale rows: {int(df['isCandidateSale'].sum()):,}",
    f"Uncategorized product transaction rows: {int(df['ProductCategory'].eq('Uncategorized').sum()):,}",
    f"UnitCost populated rows: {int(df['UnitCost'].notna().sum()):,}",
    f"StockQuantity populated rows: {int(df['StockQuantity'].notna().sum()):,}",
    f"Total SalesAmount (all rows): {df['SalesAmount'].sum():,.2f}",
    f"Source file: {RAW_FILE}", f"Cleaned dataset: {OUTPUT_FILE}",
    f"Product ID mapping: {PRODUCT_MAP_FILE}", "",
    "Important notes:",
    "- No transaction rows were deleted.",
    "- Cross-sheet duplicates are flagged, not removed, pending validation.",
    "- Within-sheet duplicate rows are flagged separately.",
    "- UnitCost and StockQuantity are unavailable in the source data.",
    "- ProductCategory is heuristic and should be reviewed.",
    "- Candidate sales are a modelling filter, not the final modelling decision.",
]
REPORT_FILE.write_text("\n".join(reportLines), encoding="utf-8")
print("\n".join(reportLines))
print("\nCleaning pipeline completed successfully.")