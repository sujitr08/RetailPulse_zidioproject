"""Smoke tests for the RetailPulse Streamlit dashboard.

Copy this file into <project-root>/tests/test_dashboard_smoke.py, then run:
    python -m pip install pytest
    python -m pytest -q tests/test_dashboard_smoke.py
"""
from pathlib import Path
import ast
import pandas as pd

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_app_has_valid_python_syntax():
    source = APP_PATH.read_text(encoding="utf-8")
    ast.parse(source, filename=str(APP_PATH))


def test_navigation_pages_are_defined():
    source = APP_PATH.read_text(encoding="utf-8")
    expected_pages = [
        "General Overview",
        "Executive Overview",
        "Sales & Forecasting",
        "Customer Segmentation",
        "Churn Risk",
        "Inventory Planning",
        "Monitoring & Model Health",
        "Data Quality & Exports",
    ]
    for page in expected_pages:
        assert page in source, f"Missing dashboard page: {page}"


def test_export_bundle_function_is_defined():
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert "build_analysis_export_bundle" in functions


def test_required_project_artifacts_exist():
    """Checks saved outputs only; it does not retrain models."""
    root = APP_PATH.parent
    candidates = {
        "cleaned transactions": root / "data/processedData/cleanedTransactions.csv",
        "daily sales": root / "data/processedData/dailySales.csv",
        "customer segments": root / "data/processedData/customerSegments.csv",
        "forecast comparison": root / "reports/modelReports/forecastModelComparison.csv",
        "churn temporal metrics": root / "reports/modelReports/churnTemporalValidationMetrics.json",
        "inventory recommendations": root / "reports/modelReports/inventoryOptimizationRecommendations.csv",
        "monitoring summary": root / "reports/modelReports/modelMonitoringSummary.json",
    }
    missing = [name for name, path in candidates.items() if not path.is_file()]
    assert not missing, "Missing saved project artifacts: " + ", ".join(missing)


def test_key_csv_artifacts_are_readable_and_nonempty():
    root = APP_PATH.parent
    paths = [
        root / "data/processedData/dailySales.csv",
        root / "data/processedData/customerSegments.csv",
        root / "reports/modelReports/forecastModelComparison.csv",
        root / "reports/modelReports/inventoryOptimizationRecommendations.csv",
    ]
    missing = [str(path.relative_to(root)) for path in paths if not path.is_file()]
    assert not missing, "Cannot validate CSVs; missing: " + ", ".join(missing)
    for path in paths:
        df = pd.read_csv(path, nrows=10)
        assert len(df.columns) > 0, f"{path.name} has no columns"
        assert not all(str(col).startswith("Unnamed:") for col in df.columns), (
            f"{path.name} appears to contain only an unnamed index column"
        )
