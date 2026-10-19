from pathlib import Path
import json
import sys
import pandas as pd
import great_expectations as gx

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "processedData" / "cleanedTransactions.csv"
REPORT_DIR = ROOT / "reports" / "modelReports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
JSON_REPORT = REPORT_DIR / "greatExpectationsValidation.json"
TEXT_REPORT = REPORT_DIR / "greatExpectationsValidation.txt"

REQUIRED_COLUMNS = [
    "invoice", "stockCode", "description", "quantity", "invoiceDate",
    "unitPrice", "customerId", "country", "sourceSheet", "ProductID",
    "ProductCategory", "UnitCost", "StockQuantity", "SalesAmount",
    "isWithinSheetDuplicate", "isCrossSheetDuplicate", "isExactDuplicate",
    "hasMissingCustomerId", "hasMissingDescription", "hasInvalidDate",
    "hasNonPositiveQuantity", "hasNonPositivePrice",
    "isCancellationInvoice", "isCandidateSale",
]

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Cleaned dataset not found: {DATA_FILE}")
df = pd.read_csv(DATA_FILE, low_memory=False)
missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
if missing_columns:
    raise ValueError(f"Required columns are missing: {missing_columns}")

# Great Expectations Core 1.x: Data Context -> Pandas Data Source -> Data Asset -> Batch.
context = gx.get_context()
try:
    data_source = context.data_sources.get("retailPulsePandas")
except Exception:
    data_source = context.data_sources.add_pandas(name="retailPulsePandas")
try:
    data_asset = data_source.get_asset("cleanedTransactions")
except Exception:
    data_asset = data_source.add_dataframe_asset(name="cleanedTransactions")
try:
    batch_definition = data_asset.get_batch_definition("wholeCleanedDataset")
except Exception:
    batch_definition = data_asset.add_batch_definition_whole_dataframe(name="wholeCleanedDataset")
batch = batch_definition.get_batch(batch_parameters={"dataframe": df})
checks = []

def run_gx_expectation(name, expectation):
    try:
        result = batch.validate(expectation)
        try:
            payload = result.to_json_dict()
        except Exception:
            payload = result.model_dump(mode="json")
        checks.append({"name": name, "success": bool(result.success),
                       "result": payload.get("result", {}), "error": None,
                       "validator": "Great Expectations"})
    except Exception as exc:
        checks.append({"name": name, "success": False, "result": {},
                       "error": f"{type(exc).__name__}: {exc}",
                       "validator": "Great Expectations"})

for column in REQUIRED_COLUMNS:
    run_gx_expectation(f"column_exists::{column}", gx.expectations.ExpectColumnToExist(column=column))
for column in ["invoice", "stockCode", "invoiceDate", "ProductID", "SalesAmount"]:
    run_gx_expectation(f"not_null::{column}", gx.expectations.ExpectColumnValuesToNotBeNull(column=column))

expected_flags = df["isWithinSheetDuplicate"].astype(bool) | df["isCrossSheetDuplicate"].astype(bool)
actual_flags = df["isExactDuplicate"].astype(bool)
flag_mismatches = int((expected_flags != actual_flags).sum())
dates = pd.to_datetime(df["invoiceDate"], errors="coerce")
checks_manual = [
    ("duplicate_flag_consistency", flag_mismatches == 0, {"mismatch_rows": flag_mismatches}),
    ("no_missing_ProductID", int(df["ProductID"].isna().sum()) == 0, {"missing_rows": int(df["ProductID"].isna().sum())}),
    ("no_missing_SalesAmount", int(df["SalesAmount"].isna().sum()) == 0, {"missing_rows": int(df["SalesAmount"].isna().sum())}),
    ("valid_invoice_dates", bool(dates.notna().all()), {"invalid_rows": int(dates.isna().sum())}),
]
for column in ["quantity", "unitPrice", "SalesAmount"]:
    parsed = pd.to_numeric(df[column], errors="coerce")
    checks_manual.append((f"numeric_values::{column}", bool(parsed.notna().all()), {"invalid_or_missing_rows": int(parsed.isna().sum())}))
for name, success, result in checks_manual:
    checks.append({"name": name, "success": bool(success), "result": result, "error": None, "validator": "Explicit data-integrity check"})

diagnostics = {
    "row_count": int(len(df)), "column_count": int(len(df.columns)),
    "missing_customerId_rows": int(df["customerId"].isna().sum()),
    "missing_description_rows": int(df["description"].isna().sum()),
    "non_positive_quantity_rows": int((pd.to_numeric(df["quantity"], errors="coerce") <= 0).sum()),
    "non_positive_unitPrice_rows": int((pd.to_numeric(df["unitPrice"], errors="coerce") <= 0).sum()),
    "within_sheet_duplicate_rows": int(df["isWithinSheetDuplicate"].astype(bool).sum()),
    "cross_sheet_duplicate_rows": int(df["isCrossSheetDuplicate"].astype(bool).sum()),
    "combined_duplicate_rows": int(df["isExactDuplicate"].astype(bool).sum()),
    "unitCost_populated_rows": int(df["UnitCost"].notna().sum()),
    "stockQuantity_populated_rows": int(df["StockQuantity"].notna().sum()),
    "candidate_sale_rows": int(df["isCandidateSale"].astype(bool).sum()),
}
failed = [item for item in checks if not item["success"]]
report = {"project": "retailPulse", "great_expectations_version": gx.__version__,
          "dataset": str(DATA_FILE), "checks_total": len(checks),
          "checks_passed": len(checks) - len(failed), "checks_failed": len(failed),
          "overall_success": len(failed) == 0, "checks": checks, "diagnostics": diagnostics,
          "interpretation": {"missing_customerId": "Known source-data limitation; reported, not fatal.",
                             "missing_description": "Known source-data limitation; reported, not fatal.",
                             "non_positive_quantity_or_price": "Reported because returns, cancellations and adjustments exist.",
                             "UnitCost_and_StockQuantity": "Expected missing until verified source values are available.",
                             "duplicates": "Flagged and reported; validation deletes no transaction rows."}}
JSON_REPORT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
lines = ["RETAILPULSE - GREAT EXPECTATIONS DATA VALIDATION", "=" * 58,
         f"Great Expectations version: {gx.__version__}", f"Rows: {len(df):,}",
         f"Columns: {len(df.columns)}", f"Checks passed: {report['checks_passed']}",
         f"Checks failed: {report['checks_failed']}", f"Overall success: {report['overall_success']}", "", "CHECK DETAILS", "-" * 58]
for item in checks:
    lines.append(f"[{'PASS' if item['success'] else 'FAIL'}] {item['name']}")
    if item["error"]:
        lines.append(f"    Error: {item['error']}")
    if item["result"]:
        lines.append(f"    Result: {json.dumps(item['result'], default=str)[:500]}")
lines += ["", "DATA QUALITY DIAGNOSTICS", "-" * 58]
for key, value in diagnostics.items():
    lines.append(f"{key}: {value:,}" if isinstance(value, int) else f"{key}: {value}")
lines += ["", "NOTES", "-" * 58,
          "- Missing customer IDs/descriptions are source-data limitations, not fatal checks.",
          "- Non-positive quantities/prices are reported, not automatically rejected.",
          "- UnitCost and StockQuantity remain missing without verified source data.",
          "- No transaction rows were deleted.", f"JSON report: {JSON_REPORT}"]
TEXT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(f"\nSaved JSON report: {JSON_REPORT}\nSaved text report: {TEXT_REPORT}")
if failed:
    print("\nValidation has failures. Review the report before proceeding.")
    sys.exit(1)
