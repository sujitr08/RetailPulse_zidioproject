from __future__ import annotations
import json
import warnings
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processedData"
REPORT_DIR = ROOT / "reports" / "modelReports"
FIGURE_DIR = ROOT / "reports" / "figures"
MODEL_DIR = ROOT / "models"

FEATURE_DAYS = 365
LABEL_DAYS = 90
TARGET_ROC_AUC = 0.88
TARGET_PRECISION_TOP_20 = 0.75


def find_dataset() -> Path:
    candidates = [
        PROCESSED_DIR / "cleanedTransactions.csv",
        PROCESSED_DIR / "cleanedData.csv",
        PROCESSED_DIR / "cleanedTransactions.parquet",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find cleanedTransactions.csv in data/processedData. "
        "Update find_dataset() if your cleaned file has a different name."
    )


def load_qualifying_sales(data_path: Path) -> pd.DataFrame:
    if data_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path, low_memory=False)

    required = {"customerId", "invoiceDate", "quantity", "unitPrice", "invoice"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Required columns missing from cleaned data: {sorted(missing)}")

    df["invoiceDate"] = pd.to_datetime(df["invoiceDate"], errors="coerce")
    df["customerId"] = pd.to_numeric(df["customerId"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["unitPrice"] = pd.to_numeric(df["unitPrice"], errors="coerce")
    if "SalesAmount" in df.columns:
        df["SalesAmount"] = pd.to_numeric(df["SalesAmount"], errors="coerce")
    else:
        df["SalesAmount"] = df["quantity"] * df["unitPrice"]

    invoice_text = df["invoice"].astype(str).str.strip().str.upper()
    sale_mask = (
        df["customerId"].notna()
        & df["invoiceDate"].notna()
        & (df["quantity"] > 0)
        & (df["unitPrice"] > 0)
        & (df["SalesAmount"] > 0)
        & ~invoice_text.str.startswith("C")
    )
    sales = df.loc[
        sale_mask,
        ["customerId", "invoiceDate", "invoice", "quantity", "unitPrice", "SalesAmount"],
    ].copy()
    sales["invoiceDate"] = pd.to_datetime(sales["invoiceDate"]).dt.tz_localize(None)
    sales["invoiceDate"] = sales["invoiceDate"].dt.normalize()
    if sales.empty:
        raise ValueError("No qualifying customer purchase rows after filtering.")
    return sales


def build_snapshot(sales: pd.DataFrame, cutoff: pd.Timestamp, max_date: pd.Timestamp):
    feature_start = cutoff - pd.Timedelta(days=FEATURE_DAYS)
    label_end = cutoff + pd.Timedelta(days=LABEL_DAYS)
    if label_end > max_date:
        raise ValueError(f"Cutoff {cutoff.date()} does not have a complete {LABEL_DAYS}-day label window.")

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
    features["featureWindowStart"] = feature_start.strftime("%Y-%m-%d")
    features["labelWindowEnd"] = label_end.strftime("%Y-%m-%d")
    return features


def make_model(y: pd.Series, seed: int = 42) -> XGBClassifier:
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
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
    )


def score_model(model, X: pd.DataFrame, y: pd.Series) -> dict:
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    top_n = max(1, int(np.ceil(len(probabilities) * 0.20)))
    top_indices = np.argsort(probabilities)[-top_n:]
    return {
        "customers": int(len(y)),
        "churn_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, probabilities)) if y.nunique() == 2 else None,
        "average_precision": float(average_precision_score(y, probabilities)),
        "precision_at_top_20_percent": float(y.iloc[top_indices].mean()),
        "precision_at_0_5_threshold": float(precision_score(y, predictions, zero_division=0)),
        "recall_at_0_5_threshold": float(recall_score(y, predictions, zero_division=0)),
        "f1_at_0_5_threshold": float(f1_score(y, predictions, zero_division=0)),
        "confusion_matrix_labels_0_1": confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y, predictions, zero_division=0, output_dict=True),
    }


def main() -> None:
    for folder in (REPORT_DIR, FIGURE_DIR, MODEL_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    data_path = find_dataset()
    print(f"Loading: {data_path}")
    sales = load_qualifying_sales(data_path)
    max_date = sales["invoiceDate"].max().normalize()

    # Cutoffs are spaced 90 days apart. The final cutoff is the latest date
    # with a complete 90-day label window. Each earlier snapshot's labels are
    # fully observed before the next snapshot's cutoff.
    final_cutoff = max_date - pd.Timedelta(days=LABEL_DAYS)
    candidate_cutoffs = [
        final_cutoff - pd.Timedelta(days=180),
        final_cutoff - pd.Timedelta(days=90),
        final_cutoff,
    ]
    snapshots = []
    for cutoff in candidate_cutoffs:
        snapshot = build_snapshot(sales, cutoff, max_date)
        snapshots.append(snapshot)
        print(
            f"Snapshot {cutoff.date()}: {len(snapshot)} customers, "
            f"churn rate {snapshot['churned'].mean():.3f}, "
            f"label end {cutoff + pd.Timedelta(days=LABEL_DAYS):%Y-%m-%d}"
        )

    # Train only on earlier snapshots whose 90-day outcomes are known before
    # the final snapshot cutoff. Test on the final snapshot. A customer may
    # appear in multiple snapshots because each row represents a different
    # point-in-time prediction; the report makes this limitation explicit.
    train_frames = snapshots[:-1]
    test_frame = snapshots[-1].copy()
    feature_columns = [
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
    train_frame = pd.concat(train_frames, axis=0)
    X_train = train_frame[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
    y_train = train_frame["churned"].astype(int)
    X_test = test_frame[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
    y_test = test_frame["churned"].astype(int)

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError("Temporal train or test data has only one churn class; inspect the selected cutoffs.")

    model = make_model(y_train)
    model.fit(X_train, y_train)
    test_metrics = score_model(model, X_test, y_test)
    test_probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (test_probabilities >= 0.5).astype(int)

    metrics = {
        "model": "XGBoostClassifier_temporal_validation",
        "label_definition": f"No qualifying purchase during the {LABEL_DAYS} days after each cutoff",
        "feature_window_days": FEATURE_DAYS,
        "label_window_days": LABEL_DAYS,
        "dataset_max_qualifying_purchase_date": str(max_date.date()),
        "training_cutoffs": [str(s["snapshotCutoff"].iloc[0]) for s in train_frames],
        "test_cutoff": str(test_frame["snapshotCutoff"].iloc[0]),
        "test_label_window_end": str(test_frame["labelWindowEnd"].iloc[0]),
        "train_snapshot_rows": int(len(train_frame)),
        "test_customers": int(len(test_frame)),
        "train_snapshot_churn_rate": float(y_train.mean()),
        "test_churn_rate": float(y_test.mean()),
        "temporal_test_metrics": {k: v for k, v in test_metrics.items() if k != "classification_report"},
        "feature_columns": feature_columns,
        "validation_method": (
            "Train on two earlier point-in-time customer snapshots and test on the latest snapshot. "
            "Earlier label windows end before the final test cutoff. Customers may recur across snapshots; "
            "this is point-in-time temporal validation, not a fully customer-disjoint test."
        ),
        "caveats": [
            "A temporal split is more realistic than a random split, but only one final cutoff is tested.",
            "Customers may occur in multiple snapshots; interpret this as repeated point-in-time scoring.",
            "The dataset is historical and ends in 2011; results may not generalize to current retail behavior.",
            "ROC-AUC and precision targets are goals, not guaranteed outcomes.",
            "Unit cost and inventory fields are not used because source values are unavailable or unverified.",
        ],
    }

    # Persist temporal evaluation and model outputs without overwriting the
    # original baseline metrics/report.
    temporal_metrics_path = REPORT_DIR / "churnTemporalValidationMetrics.json"
    temporal_metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    predictions_out = test_frame[["churned"]].copy()
    predictions_out["churnProbability"] = test_probabilities
    predictions_out["predictedChurnAt0_5"] = predictions
    predictions_out["riskBand"] = pd.cut(
        predictions_out["churnProbability"],
        bins=[-0.001, 0.3, 0.6, 1.001],
        labels=["Low", "Medium", "High"],
    )
    predictions_out.index.name = "customerId"
    predictions_out.reset_index().sort_values(
        "churnProbability", ascending=False
    ).to_csv(REPORT_DIR / "churnTemporalRiskPredictions.csv", index=False)

    importance = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(REPORT_DIR / "churnTemporalFeatureImportance.csv", index=False)

    plt.figure(figsize=(9, 6))
    plt.barh(importance.head(10).sort_values("importance")["feature"],
             importance.head(10).sort_values("importance")["importance"])
    plt.title("Temporal Churn Model — Feature Importance")
    plt.xlabel("XGBoost importance")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "churnTemporalFeatureImportance.png", dpi=160)
    plt.close()

    model_path = MODEL_DIR / "churnXgboostTemporalModel.json"
    model.save_model(model_path)
    joblib.dump(
        {
            "featureColumns": feature_columns,
            "testCutoff": metrics["test_cutoff"],
            "modelType": "XGBoostClassifier_temporal_validation",
        },
        MODEL_DIR / "churnTemporalModelMetadata.joblib",
    )

    report_lines = [
        "RETAILPULSE — CUSTOMER CHURN TEMPORAL VALIDATION",
        "=" * 54,
        f"Dataset: {data_path}",
        f"Maximum qualifying purchase date: {max_date.date()}",
        f"Feature window: {FEATURE_DAYS} days",
        f"Label window: {LABEL_DAYS} days",
        f"Training cutoffs: {', '.join(metrics['training_cutoffs'])}",
        f"Test cutoff: {metrics['test_cutoff']}",
        f"Test label window end: {metrics['test_label_window_end']}",
        f"Training snapshot rows: {len(train_frame):,}",
        f"Test customers: {len(test_frame):,}",
        "",
        f"Temporal ROC-AUC: {test_metrics['roc_auc']:.4f} (target >= {TARGET_ROC_AUC:.2f})",
        f"Temporal precision@top 20%: {test_metrics['precision_at_top_20_percent']:.4f} (target >= {TARGET_PRECISION_TOP_20:.2f})",
        f"Temporal average precision: {test_metrics['average_precision']:.4f}",
        f"Precision at 0.5: {test_metrics['precision_at_0_5_threshold']:.4f}",
        f"Recall at 0.5: {test_metrics['recall_at_0_5_threshold']:.4f}",
        f"F1 at 0.5: {test_metrics['f1_at_0_5_threshold']:.4f}",
        "",
        "CAVEATS",
        *[f"- {note}" for note in metrics["caveats"]],
        "",
        "Classification report (temporal test snapshot):",
        classification_report(y_test, predictions, zero_division=0),
        "",
        f"Metrics JSON: {temporal_metrics_path}",
        f"Predictions CSV: {REPORT_DIR / 'churnTemporalRiskPredictions.csv'}",
    ]
    report_path = REPORT_DIR / "churnTemporalValidationReport.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\nCHURN TEMPORAL VALIDATION RESULTS")
    print("=" * 42)
    print(f"Training cutoffs: {', '.join(metrics['training_cutoffs'])}")
    print(f"Test cutoff: {metrics['test_cutoff']}")
    print(f"ROC-AUC: {test_metrics['roc_auc']:.4f} (target >= {TARGET_ROC_AUC:.2f})")
    print(
        f"Precision@top 20%: {test_metrics['precision_at_top_20_percent']:.4f} "
        f"(target >= {TARGET_PRECISION_TOP_20:.2f})"
    )
    print(f"Metrics: {temporal_metrics_path}")
    print(f"Report: {report_path}")
    print("This temporal evaluation is a stronger check, not proof of production performance.")


if __name__ == "__main__":
    main()
