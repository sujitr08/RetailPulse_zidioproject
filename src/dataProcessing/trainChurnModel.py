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
    roc_auc_score, average_precision_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processedData"
REPORT_DIR = ROOT / "reports" / "modelReports"
FIGURE_DIR = ROOT / "reports" / "figures"
MODEL_DIR = ROOT / "models"


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


def main() -> None:
    for folder in (REPORT_DIR, FIGURE_DIR, MODEL_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    data_path = find_dataset()
    print(f"Loading: {data_path}")
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

    # A qualifying purchase excludes returns/cancellations and non-positive sales.
    invoice_text = df["invoice"].astype(str).str.strip().str.upper()
    sale_mask = (
        df["customerId"].notna()
        & df["invoiceDate"].notna()
        & (df["quantity"] > 0)
        & (df["unitPrice"] > 0)
        & (df["SalesAmount"] > 0)
        & ~invoice_text.str.startswith("C")
    )
    sales = df.loc[sale_mask, ["customerId", "invoiceDate", "invoice", "quantity", "unitPrice", "SalesAmount"]].copy()
    if sales.empty:
        raise ValueError("No qualifying customer purchase rows after filtering.")

    max_date = sales["invoiceDate"].max().normalize()
    label_days = 90
    feature_days = 365
    cutoff = max_date - pd.Timedelta(days=label_days)
    feature_start = cutoff - pd.Timedelta(days=feature_days)
    feature_sales = sales[(sales["invoiceDate"] > feature_start) & (sales["invoiceDate"] <= cutoff)].copy()
    label_sales = sales[(sales["invoiceDate"] > cutoff) & (sales["invoiceDate"] <= max_date)].copy()
    if feature_sales.empty or label_sales.empty:
        raise ValueError("Insufficient data in feature/label windows; inspect date coverage.")

    # Customer features are calculated strictly before/at cutoff.
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
    features["recencyDays"] = (cutoff - features["lastPurchaseDate"].dt.normalize()).dt.days
    features["uniquePurchaseDays"] = feature_sales.groupby("customerId")["invoiceDate"].apply(lambda s: s.dt.normalize().nunique())
    features["averageQuantity"] = feature_sales.groupby("customerId")["quantity"].mean()
    features["purchaseSpanDays"] = (
        feature_sales.groupby("customerId")["invoiceDate"].max().dt.normalize()
        - feature_sales.groupby("customerId")["invoiceDate"].min().dt.normalize()
    ).dt.days
    label_customers = set(label_sales["customerId"].unique())
    features["churned"] = (~features.index.isin(label_customers)).astype(int)
    features = features.replace([np.inf, -np.inf], np.nan)
    feature_columns = [c for c in features.columns if c not in ("lastPurchaseDate", "churned")]
    X = features[feature_columns].fillna(0).astype(float)
    y = features["churned"].astype(int)

    if y.nunique() < 2:
        raise ValueError("Only one churn class found; cannot train a binary classifier.")
    if y.value_counts().min() < 2:
        raise ValueError("Too few examples in one churn class for a stratified split.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    negative, positive = np.bincount(y_train, minlength=2)
    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=float(negative / max(positive, 1)),
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    print(f"Customers: {len(X)} | churn rate: {y.mean():.3f} | cutoff: {cutoff.date()} | label end: {max_date.date()}")
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    top_n = max(1, int(np.ceil(len(probabilities) * 0.20)))
    top_indices = np.argsort(probabilities)[-top_n:]
    precision_top20 = float(y_test.iloc[top_indices].mean())

    metrics = {
        "model": "XGBoostClassifier",
        "label_definition": f"No qualifying purchase during the {label_days} days after cutoff",
        "feature_window_days": feature_days,
        "label_window_days": label_days,
        "observation_cutoff": str(cutoff.date()),
        "label_window_end": str(max_date.date()),
        "customers_total": int(len(features)),
        "train_customers": int(len(X_train)),
        "test_customers": int(len(X_test)),
        "churn_rate_all": float(y.mean()),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "precision_at_top_20_percent": precision_top20,
        "precision_at_0_5_threshold": float(precision_score(y_test, predictions, zero_division=0)),
        "recall_at_0_5_threshold": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_at_0_5_threshold": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix_labels_0_1": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "feature_columns": feature_columns,
        "note": "Retrospective model. A random stratified split is used for initial baseline only; use a time-based customer validation before production claims. Precision@top20 is calculated on the held-out test set.",
    }

    model_path = MODEL_DIR / "churnXgboostModel.json"
    model.save_model(model_path)
    joblib.dump({"featureColumns": feature_columns, "cutoff": str(cutoff.date()), "modelType": "XGBoostClassifier"}, MODEL_DIR / "churnModelMetadata.joblib")
    metrics_path = REPORT_DIR / "churnModelMetrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    output = features[["churned"]].copy()
    output["churnProbability"] = model.predict_proba(X)[:, 1]
    output["riskBand"] = pd.cut(output["churnProbability"], bins=[-0.001, 0.3, 0.6, 1.001], labels=["Low", "Medium", "High"])
    output.index.name = "customerId"
    output.reset_index().sort_values("churnProbability", ascending=False).to_csv(REPORT_DIR / "churnRiskPredictions.csv", index=False)

    importance = pd.DataFrame({"feature": feature_columns, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    importance.to_csv(REPORT_DIR / "churnFeatureImportance.csv", index=False)
    plt.figure(figsize=(9, 6))
    plot_data = importance.head(12).sort_values("importance")
    plt.barh(plot_data["feature"], plot_data["importance"])
    plt.title("XGBoost Churn — Top Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "churnFeatureImportance.png", dpi=160)
    plt.close()

    # SHAP is optional at runtime: save explanation plot and values when compatible.
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        sample = X_test.sample(n=min(500, len(X_test)), random_state=42)
        shap_values = explainer.shap_values(sample)
        plt.figure()
        shap.summary_plot(shap_values, sample, show=False, max_display=12)
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "churnShapSummary.png", dpi=160, bbox_inches="tight")
        plt.close()
        shap_importance = pd.DataFrame({"feature": sample.columns, "meanAbsoluteShap": np.abs(shap_values).mean(axis=0)}).sort_values("meanAbsoluteShap", ascending=False)
        shap_importance.to_csv(REPORT_DIR / "churnShapImportance.csv", index=False)
        metrics["shap_status"] = "success"
    except Exception as exc:
        metrics["shap_status"] = f"failed: {type(exc).__name__}: {exc}"
        warnings.warn("XGBoost trained, but SHAP explanation failed. See metrics JSON for details.")

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report_lines = [
        "RETAILPULSE — CUSTOMER CHURN MODEL REPORT",
        "=" * 48,
        f"Dataset: {data_path}",
        f"Observation cutoff: {cutoff.date()}",
        f"Feature window: {feature_start.date()} through {cutoff.date()} (365 days)",
        f"Churn label window: after {cutoff.date()} through {max_date.date()} (90 days)",
        f"Customers: {len(features):,}",
        f"Churn rate: {y.mean():.2%}",
        "",
        f"ROC-AUC: {metrics['roc_auc']:.4f} (target >= 0.88)",
        f"Precision@top 20%: {precision_top20:.4f} (target >= 0.75)",
        f"Average precision: {metrics['average_precision']:.4f}",
        f"Precision at 0.5: {metrics['precision_at_0_5_threshold']:.4f}",
        f"Recall at 0.5: {metrics['recall_at_0_5_threshold']:.4f}",
        f"SHAP: {metrics.get('shap_status', 'not run')}",
        "",
        "CAVEATS",
        "- Churn is operationally defined as no qualifying purchase in the next 90 days.",
        "- This is an initial baseline using a stratified random split; time-based validation is required before production claims.",
        "- Inventory and unit cost fields are not used because source values are unavailable/unverified.",
        "- The target metrics are goals, not guaranteed outcomes.",
        "",
        "Classification report (test set):",
        classification_report(y_test, predictions, zero_division=0),
    ]
    (REPORT_DIR / "churnModelReport.txt").write_text("\n".join(report_lines), encoding="utf-8")

    print("\nCHURN MODEL RESULTS")
    print("=" * 48)
    print(f"ROC-AUC: {metrics['roc_auc']:.4f} (target >= 0.88)")
    print(f"Precision@top 20%: {precision_top20:.4f} (target >= 0.75)")
    print(f"SHAP status: {metrics.get('shap_status', 'not run')}")
    print(f"Metrics: {metrics_path}")
    print(f"Report: {REPORT_DIR / 'churnModelReport.txt'}")
    print("This is a baseline; do not claim targets are met until metrics support them.")


if __name__ == "__main__":
    main()
