from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processedData"
REPORT_DIR = ROOT / "reports" / "modelReports"
FEATURE_DAYS = 365
LABEL_DAYS = 90
TARGET_ROC_AUC = 0.88
TARGET_PRECISION_TOP_20 = 0.75

FEATURE_COLUMNS = [
    "frequency",
    "quantityTotal",
    "monetaryTotal",
    "averageOrderValue",
    "averageUnitPrice",
    "transactionRows",
    "recencyDays",
    "uniquePurchaseDays",
    "averageQuantity",
    "purchaseSpanDays",
]


def load_sales() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = PROCESSED_DIR / "cleanedTransactions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    required = {
        "customerId", "invoiceDate", "invoice", "quantity", "unitPrice",
        "SalesAmount", "isCrossSheetDuplicate",
    }
    df = pd.read_csv(path, low_memory=False)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Required columns missing from cleaned dataset: {sorted(missing)}")

    df["invoiceDate"] = pd.to_datetime(df["invoiceDate"], errors="coerce")
    for column in ("customerId", "quantity", "unitPrice", "SalesAmount"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    cross_flag = (
        df["isCrossSheetDuplicate"].astype(str).str.strip().str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )
    if cross_flag.isna().any():
        raise ValueError("isCrossSheetDuplicate contains missing or unrecognized values.")
    df["isCrossSheetDuplicate"] = cross_flag.astype(bool)

    invoice_text = df["invoice"].astype(str).str.strip().str.upper()
    sale_mask = (
        df["customerId"].notna()
        & df["invoiceDate"].notna()
        & (df["quantity"] > 0)
        & (df["unitPrice"] > 0)
        & (df["SalesAmount"] > 0)
        & ~invoice_text.str.startswith("C")
    )

    columns = [
        "customerId", "invoiceDate", "invoice", "quantity", "unitPrice", "SalesAmount"
    ]
    baseline = df.loc[sale_mask, columns].copy()
    filtered = df.loc[sale_mask & ~df["isCrossSheetDuplicate"], columns].copy()

    for sales in (baseline, filtered):
        sales["invoiceDate"] = sales["invoiceDate"].dt.tz_localize(None).dt.normalize()

    if baseline.empty or filtered.empty:
        raise ValueError("No qualifying sales remain for one or both experiments.")

    print(f"Source rows: {len(df):,}")
    print(f"Qualifying sale rows, baseline: {len(baseline):,}")
    print(f"Qualifying sale rows, excluding cross-sheet copies: {len(filtered):,}")
    print(
        "Cross-sheet duplicate rows flagged in source: "
        f"{int(df['isCrossSheetDuplicate'].sum()):,}"
    )
    return baseline, filtered


def build_snapshot(
    sales: pd.DataFrame, cutoff: pd.Timestamp, max_date: pd.Timestamp
) -> pd.DataFrame:
    feature_start = cutoff - pd.Timedelta(days=FEATURE_DAYS)
    label_end = cutoff + pd.Timedelta(days=LABEL_DAYS)
    if label_end > max_date:
        raise ValueError(
            f"Cutoff {cutoff.date()} does not have a complete {LABEL_DAYS}-day label window."
        )

    feature_sales = sales[
        (sales["invoiceDate"] > feature_start) & (sales["invoiceDate"] <= cutoff)
    ].copy()
    label_sales = sales[
        (sales["invoiceDate"] > cutoff) & (sales["invoiceDate"] <= label_end)
    ].copy()

    if feature_sales.empty:
        raise ValueError(f"No feature-window sales for cutoff {cutoff.date()}.")

    grouped = feature_sales.groupby("customerId", sort=False)
    features = grouped.agg(
        lastPurchaseDate=("invoiceDate", "max"),
        frequency=("invoice", "nunique"),
        quantityTotal=("quantity", "sum"),
        monetaryTotal=("SalesAmount", "sum"),
        averageOrderValue=("SalesAmount", "mean"),
        averageUnitPrice=("unitPrice", "mean"),
        transactionRows=("invoice", "size"),
    )
    features["recencyDays"] = (
        cutoff - features["lastPurchaseDate"].dt.normalize()
    ).dt.days
    features["uniquePurchaseDays"] = feature_sales.groupby("customerId")[
        "invoiceDate"
    ].nunique()
    features["averageQuantity"] = feature_sales.groupby("customerId")["quantity"].mean()
    features["purchaseSpanDays"] = (
        feature_sales.groupby("customerId")["invoiceDate"].max()
        - feature_sales.groupby("customerId")["invoiceDate"].min()
    ).dt.days

    label_customers = set(label_sales["customerId"].unique())
    features["churned"] = (~features.index.isin(label_customers)).astype(int)
    features["snapshotCutoff"] = cutoff.strftime("%Y-%m-%d")
    features["labelWindowEnd"] = label_end.strftime("%Y-%m-%d")
    return features


def make_model(y: pd.Series) -> XGBClassifier:
    counts = np.bincount(y.astype(int), minlength=2)
    negative, positive = int(counts[0]), int(counts[1])
    return XGBClassifier(
        n_estimators=450,
        max_depth=3,
        learning_rate=0.035,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        reg_alpha=0.05,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=float(negative / max(positive, 1)),
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )


def score_model(model: XGBClassifier, frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    X = frame[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
    y = frame["churned"].astype(int)
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    top_n = max(1, int(np.ceil(len(probabilities) * 0.20)))
    top_indices = np.argsort(probabilities)[-top_n:]

    metrics = {
        "customers": int(len(y)),
        "churn_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, probabilities)) if y.nunique() == 2 else None,
        "average_precision": float(average_precision_score(y, probabilities)),
        "precision_at_top_20_percent": float(y.iloc[top_indices].mean()),
        "precision_at_0_5_threshold": float(precision_score(y, predictions, zero_division=0)),
        "recall_at_0_5_threshold": float(recall_score(y, predictions, zero_division=0)),
        "f1_at_0_5_threshold": float(f1_score(y, predictions, zero_division=0)),
        "confusion_matrix_labels_0_1": confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y, predictions, zero_division=0, output_dict=True
        ),
    }

    predictions_out = pd.DataFrame(
        {
            "customerId": frame.index,
            "actualChurn": y.to_numpy(),
            "churnProbability": probabilities,
            "predictedChurnAt0_5": predictions,
        }
    ).sort_values("churnProbability", ascending=False)
    return metrics, predictions_out


def run_experiment(
    name: str, sales: pd.DataFrame, cutoffs: list[pd.Timestamp], max_date: pd.Timestamp
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    snapshots = [build_snapshot(sales, cutoff, max_date) for cutoff in cutoffs]
    train_frame = pd.concat(snapshots[:-1], axis=0)
    test_frame = snapshots[-1].copy()

    X_train = (
        train_frame[FEATURE_COLUMNS]
        .replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
    )
    y_train = train_frame["churned"].astype(int)
    if y_train.nunique() < 2 or test_frame["churned"].nunique() < 2:
        raise ValueError(f"{name}: training or test split has only one churn class.")

    model = make_model(y_train)
    model.fit(X_train, y_train)
    metrics, predictions = score_model(model, test_frame)
    result = {
        "experiment": name,
        "train_snapshot_rows": int(len(train_frame)),
        "test_customers": int(len(test_frame)),
        "training_cutoffs": [str(s["snapshotCutoff"].iloc[0]) for s in snapshots[:-1]],
        "test_cutoff": str(test_frame["snapshotCutoff"].iloc[0]),
        "test_label_window_end": str(test_frame["labelWindowEnd"].iloc[0]),
        "train_churn_rate": float(y_train.mean()),
        "test_metrics": metrics,
        "feature_importance": dict(
            sorted(
                zip(FEATURE_COLUMNS, model.feature_importances_.tolist()),
                key=lambda item: item[1],
                reverse=True,
            )
        ),
    }
    predictions.insert(0, "experiment", name)
    return result, predictions, test_frame


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    baseline_sales, filtered_sales = load_sales()

    # The same date cutoffs are used for both experiments. The latest date is
    # retained because the excluded copies are older-sheet records.
    max_date = baseline_sales["invoiceDate"].max().normalize()
    final_cutoff = max_date - pd.Timedelta(days=LABEL_DAYS)
    cutoffs = [
        final_cutoff - pd.Timedelta(days=180),
        final_cutoff - pd.Timedelta(days=90),
        final_cutoff,
    ]
    print("\nCutoffs:", ", ".join(str(c.date()) for c in cutoffs))

    baseline_result, baseline_predictions, _ = run_experiment(
        "baseline_all_qualifying_rows", baseline_sales, cutoffs, max_date
    )
    filtered_result, filtered_predictions, _ = run_experiment(
        "exclude_cross_sheet_duplicate_copies", filtered_sales, cutoffs, max_date
    )

    results = {
        "title": "RetailPulse churn duplicate-policy sensitivity experiment",
        "dataset": "data/processedData/cleanedTransactions.csv",
        "feature_window_days": FEATURE_DAYS,
        "label_window_days": LABEL_DAYS,
        "cutoffs": [str(c.date()) for c in cutoffs],
        "duplicate_policy": (
            "Baseline uses all qualifying rows. Filtered experiment excludes only rows "
            "flagged isCrossSheetDuplicate=True. Within-sheet duplicates are retained. "
            "The original cleaned dataset is not modified."
        ),
        "caveats": [
            "This is a sensitivity experiment, not proof that all flagged copies are invalid.",
            "Only one final temporal test cutoff is evaluated.",
            "Customers may recur across point-in-time snapshots.",
            "Historical data ends in 2011 and may not generalize to current retail behavior.",
            "Acceptance thresholds are goals, not guaranteed outcomes.",
        ],
        "baseline": baseline_result,
        "cross_sheet_filtered": filtered_result,
    }

    metrics_path = REPORT_DIR / "churnDuplicatePolicyComparison.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    comparison_rows = []
    for result in (baseline_result, filtered_result):
        metrics = result["test_metrics"]
        comparison_rows.append(
            {
                "experiment": result["experiment"],
                "train_snapshot_rows": result["train_snapshot_rows"],
                "test_customers": result["test_customers"],
                "train_churn_rate": result["train_churn_rate"],
                "test_churn_rate": metrics["churn_rate"],
                "roc_auc": metrics["roc_auc"],
                "target_roc_auc": TARGET_ROC_AUC,
                "precision_at_top_20_percent": metrics["precision_at_top_20_percent"],
                "target_precision_at_top_20_percent": TARGET_PRECISION_TOP_20,
                "average_precision": metrics["average_precision"],
                "precision_at_0_5_threshold": metrics["precision_at_0_5_threshold"],
                "recall_at_0_5_threshold": metrics["recall_at_0_5_threshold"],
                "f1_at_0_5_threshold": metrics["f1_at_0_5_threshold"],
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    comparison_path = REPORT_DIR / "churnDuplicatePolicyComparison.csv"
    comparison.to_csv(comparison_path, index=False)

    predictions_path = REPORT_DIR / "churnDuplicatePolicyPredictions.csv"
    pd.concat([baseline_predictions, filtered_predictions], ignore_index=True).to_csv(
        predictions_path, index=False
    )

    report_lines = [
        "RETAILPULSE - CHURN DUPLICATE-POLICY SENSITIVITY EXPERIMENT",
        "=" * 66,
        f"Dataset: {PROCESSED_DIR / 'cleanedTransactions.csv'}",
        f"Cutoffs: {', '.join(str(c.date()) for c in cutoffs)}",
        f"Feature window: {FEATURE_DAYS} days",
        f"Label window: {LABEL_DAYS} days",
        "",
        "COMPARISON",
        comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "",
        "INTERPRETATION",
        "- Baseline retains all qualifying rows.",
        "- Filtered experiment excludes only isCrossSheetDuplicate=True rows.",
        "- Within-sheet duplicate rows remain in both experiments.",
        "- Compare both models on the same final temporal cutoff.",
        "- This experiment does not prove which duplicate policy is definitively correct.",
        "",
        "CAVEATS",
        *[f"- {note}" for note in results["caveats"]],
        "",
        f"Metrics JSON: {metrics_path}",
        f"Comparison CSV: {comparison_path}",
        f"Predictions CSV: {predictions_path}",
    ]
    report_path = REPORT_DIR / "churnDuplicatePolicyComparison.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\nCHURN DUPLICATE-POLICY COMPARISON")
    print("=" * 42)
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved: {metrics_path}")
    print(f"Saved: {comparison_path}")
    print(f"Saved: {predictions_path}")
    print(f"Saved: {report_path}")
    print("Original cleaned data and original churn reports were not modified.")


if __name__ == "__main__":
    main()
