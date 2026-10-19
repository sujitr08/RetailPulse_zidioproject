from pathlib import Path
import json
import math
import warnings
import io
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processedData"
REPORT_DIR = ROOT / "reports" / "modelReports"
FIGURE_DIR = ROOT / "reports" / "figures"

st.set_page_config(
    page_title="RetailPulse | Customer & Demand Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    .hero {padding: 1.1rem 1.3rem; border-radius: 14px;
           border: 1px solid rgba(128,128,128,.25); margin-bottom: 1rem;}
    .small-note {font-size: .88rem; opacity: .8;}
    div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.2);
        padding: .8rem; border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def read_csv_if_exists(path_string, **kwargs):
    path = Path(path_string)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as exc:
        st.warning(f"Could not read `{path.name}`: {exc}")
        return None


@st.cache_data(show_spinner=False)
def read_json_if_exists(path_string):
    path = Path(path_string)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def first_existing(*paths):
    for path in paths:
        if path.exists():
            return path
    return None


def load_csv_candidates(*relative_paths):
    path = first_existing(*(ROOT / p for p in relative_paths))
    if path is None:
        return None, None
    return read_csv_if_exists(str(path)), path


def find_column(df, candidates):
    if df is None:
        return None
    lower_map = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    for col in df.columns:
        normalized = str(col).lower().replace("_", "").replace(" ", "")
        for candidate in candidates:
            wanted = candidate.lower().replace("_", "").replace(" ", "")
            if normalized == wanted:
                return col
    return None


def as_numeric(df, col):
    return pd.to_numeric(df[col], errors="coerce") if df is not None and col else None


def show_missing_file(path_hint, reason="This page needs an output file from an earlier pipeline step."):
    st.info(f"{reason}\n\nExpected file: `{path_hint}`")


def download_csv_button(df, filename, label="Download CSV"):
    if df is not None and not df.empty:
        st.download_button(
            label=label,
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=filename,
            mime="text/csv",
        )


def metric_or_na(label, value, help_text=None):
    st.metric(label, value if value is not None else "N/A", help=help_text)


def read_text(path):
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return "Report could not be read as UTF-8."


def build_analysis_export_bundle(root):
    """Build a ZIP in memory containing available analysis outputs (excluding the large raw/cleaned data)."""
    candidate_paths = [
        "data/processedData/dailySales.csv",
        "data/processedData/customerSegments.csv",
        "data/processedData/prophetHoldoutForecast.csv",
        "data/processedData/lstmHoldoutForecast.csv",
        "data/processedData/ensembleHoldoutForecast.csv",
        "reports/modelReports/customerSegments.csv",
        "reports/modelReports/prophetHoldoutForecast.csv",
        "reports/modelReports/lstmHoldoutForecast.csv",
        "reports/modelReports/ensembleHoldoutForecast.csv",
        "reports/modelReports/forecastModelComparison.csv",
        "reports/modelReports/forecastEnsembleMetrics.json",
        "reports/modelReports/prophetBaselineMetrics.json",
        "reports/modelReports/lstmBaselineMetrics.json",
        "reports/modelReports/churnTemporalRiskPredictions.csv",
        "reports/modelReports/churnRiskPredictions.csv",
        "reports/modelReports/churnTemporalValidationMetrics.json",
        "reports/modelReports/churnTemporalValidationReport.txt",
        "reports/modelReports/churnShapImportance.csv",
        "reports/modelReports/churnTemporalFeatureImportance.csv",
        "reports/modelReports/inventoryOptimizationRecommendations.csv",
        "reports/modelReports/inventoryOptimizationSummary.json",
        "reports/modelReports/inventoryDemandConfidenceAudit.csv",
        "reports/modelReports/inventoryDemandAudit.txt",
        "reports/modelReports/edaReport.txt",
        "reports/modelReports/greatExpectationsValidation.txt",
        "reports/modelReports/modelMonitoringSummary.json",
        "reports/modelReports/forecastMonitoringMetrics.csv",
        "reports/modelReports/churnMonitoringMetrics.json",
        "reports/modelReports/salesDataDriftReport.html",
        "reports/modelReports/churnScoreDriftReport.html",
        "reports/modelReports/modelMonitoringReport.txt",
    ]
    buffer = io.BytesIO()
    added = []
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in candidate_paths:
            source = root / relative
            if source.is_file():
                archive.write(source, arcname=relative.replace("\\", "/"))
                added.append(relative)
        readme = (
            "RetailPulse analysis export bundle\n"
            "==================================\n\n"
            "This bundle includes available saved analysis outputs and reports.\n"
            "Missing artifacts are omitted. The large cleaned transaction dataset is\n"
            "excluded intentionally; download it separately from Data Quality & Exports.\n\n"
            "Interpretation notes:\n"
            "- Forecast and churn metrics are saved evaluation results, not guarantees.\n"
            "- Monitoring files are offline snapshots, not live alerts.\n"
            "- Inventory outputs are historical estimates; stock-on-hand and unit costs\n"
            "  are not available, so do not interpret them as confirmed savings or orders.\n"
        )
        archive.writestr("README_export_notes.txt", readme)
        archive.writestr("included_files.txt", "\n".join(added) + "\n")
    buffer.seek(0)
    return buffer.getvalue(), added


# ---------- Load existing project artifacts ----------
cleaned, cleaned_path = load_csv_candidates("data/processedData/cleanedTransactions.csv")
daily_sales, daily_path = load_csv_candidates("data/processedData/dailySales.csv")
segments, segments_path = load_csv_candidates(
    "data/processedData/customerSegments.csv",
    "reports/modelReports/customerSegments.csv",
)
prophet, prophet_path = load_csv_candidates(
    "data/processedData/prophetHoldoutForecast.csv",
    "reports/modelReports/prophetHoldoutForecast.csv",
)
lstm, lstm_path = load_csv_candidates(
    "data/processedData/lstmHoldoutForecast.csv",
    "reports/modelReports/lstmHoldoutForecast.csv",
)
ensemble, ensemble_path = load_csv_candidates(
    "data/processedData/ensembleHoldoutForecast.csv",
    "reports/modelReports/ensembleHoldoutForecast.csv",
)
inventory, inventory_path = load_csv_candidates(
    "reports/modelReports/inventoryOptimizationRecommendations.csv",
)
churn, churn_path = load_csv_candidates(
    "reports/modelReports/churnTemporalRiskPredictions.csv",
    "reports/modelReports/churnRiskPredictions.csv",
)
forecast_metrics = {
    "Prophet": read_json_if_exists(str(REPORT_DIR / "prophetBaselineMetrics.json")),
    "LSTM": read_json_if_exists(str(REPORT_DIR / "lstmBaselineMetrics.json")),
    "Ensemble": read_json_if_exists(str(REPORT_DIR / "forecastEnsembleMetrics.json")),
}
churn_metrics = read_json_if_exists(str(REPORT_DIR / "churnTemporalValidationMetrics.json"))
churn_shap, churn_shap_path = load_csv_candidates(
    "reports/modelReports/churnShapImportance.csv",
    "reports/modelReports/churnTemporalFeatureImportance.csv",
    "reports/modelReports/churnFeatureImportance.csv",
)
inventory_confidence, inventory_confidence_path = load_csv_candidates(
    "reports/modelReports/inventoryDemandConfidenceAudit.csv",
)
tracker_path = first_existing(ROOT / "projectTracker.md", ROOT / "projectTracker_reviewed.md")
inventory_summary = read_json_if_exists(str(REPORT_DIR / "inventoryOptimizationSummary.json"))
eda_report_path = REPORT_DIR / "edaReport.txt"
segmentation_report_path = REPORT_DIR / "customerSegmentationReport.txt"
churn_report_path = REPORT_DIR / "churnTemporalValidationReport.txt"
inventory_report_path = REPORT_DIR / "inventoryDemandAudit.txt"

# ---------- Sidebar navigation ----------
st.sidebar.title("RetailPulse")
st.sidebar.caption("AI-Powered Customer Analytics & Demand Forecasting")
page = st.sidebar.radio(
    "Navigate",
    [
        "General Overview",
        "Executive Overview",
        "Sales & Forecasting",
        "Customer Segmentation",
        "Churn Risk",
        "Inventory Planning",
        "Monitoring & Model Health",
        "Data Quality & Exports",
    ],
)
st.sidebar.divider()
st.sidebar.caption("Data freshness")
if daily_sales is not None and "date" in daily_sales.columns:
    st.sidebar.write(f"Sales history ends: **{daily_sales['date'].max()}**")
else:
    st.sidebar.write("Historical dataset")
st.sidebar.caption("This is a historical demonstration, not a live ERP/inventory feed.")


# ---------- Page 1: Overview ----------
if page == "Executive Overview":
    st.markdown(
        """
        <div class="hero">
        <h1>RetailPulse</h1>
        <p>Turn historical retail transactions into demand insights, customer segments,
        churn-risk signals, and inventory-planning recommendations.</p>
        <p class="small-note">Model outputs are evaluated baselines. Forecast accuracy and
        churn targets have not all been achieved; inventory and customer data are historical.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if cleaned is not None:
        transaction_count = len(cleaned)
        product_col = find_column(cleaned, ["ProductID"])
        customer_col = find_column(cleaned, ["customerId", "Customer ID"])
        date_col = find_column(cleaned, ["invoiceDate", "InvoiceDate"])
        product_count = cleaned[product_col].nunique() if product_col else None
        customer_count = cleaned[customer_col].nunique(dropna=True) if customer_col else None
        date_series = pd.to_datetime(cleaned[date_col], errors="coerce") if date_col else None
    else:
        transaction_count, product_count, customer_count, date_series = None, None, None, None

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        metric_or_na("Transaction rows", f"{transaction_count:,}" if transaction_count is not None else None)
    with k2:
        metric_or_na("Distinct products", f"{product_count:,}" if product_count is not None else None)
    with k3:
        metric_or_na("Identified customers", f"{customer_count:,}" if customer_count is not None else None,
                     "Customers with an available customer ID.")
    with k4:
        metric_or_na("Segmented customers", f"{len(segments):,}" if segments is not None else None)

    st.subheader("Business intelligence at a glance")
    left, right = st.columns(2)
    with left:
        if daily_sales is not None:
            date_col = find_column(daily_sales, ["date", "ds"])
            sales_col = find_column(daily_sales, ["dailySales", "y", "sales"])
            if date_col and sales_col:
                chart_df = daily_sales.copy()
                chart_df[date_col] = pd.to_datetime(chart_df[date_col], errors="coerce")
                chart_df[sales_col] = pd.to_numeric(chart_df[sales_col], errors="coerce")
                chart_df = chart_df.dropna(subset=[date_col, sales_col])
                st.plotly_chart(
                    px.line(chart_df, x=date_col, y=sales_col, title="Historical daily sales",
                            labels={date_col: "Date", sales_col: "Sales"}),
                    use_container_width=True,
                )
            else:
                show_missing_file("data/processedData/dailySales.csv", "Daily sales columns were not recognized.")
        else:
            show_missing_file("data/processedData/dailySales.csv")
    with right:
        if segments is not None:
            cluster_col = find_column(segments, ["cluster", "kmeansCluster", "segment", "clusterLabel"])
            if cluster_col:
                count_df = segments[cluster_col].fillna("Unknown").astype(str).value_counts().rename_axis("Segment").reset_index(name="Customers")
                st.plotly_chart(px.bar(count_df, x="Segment", y="Customers", title="Customer segment sizes"),
                                use_container_width=True)
            else:
                st.info("Customer segment file exists, but the cluster column name was not recognized.")
        else:
            show_missing_file("data/processedData/customerSegments.csv")

    st.subheader("Model health snapshot")
    metric_cols = st.columns(3)
    with metric_cols[0]:
        m = forecast_metrics.get("Prophet") or {}
        val = m.get("mape_nonzero", m.get("mape", m.get("MAPE")))
        metric_or_na("Prophet MAPE", f"{float(val):.2f}%" if isinstance(val, (int, float)) and val > 1 else
                     (f"{float(val) * 100:.2f}%" if isinstance(val, (int, float)) else None),
                     "Latest recorded holdout metric; nonzero-day MAPE may be used.")
    with metric_cols[1]:
        m = churn_metrics or {}
        val = m.get("roc_auc", m.get("ROC-AUC", m.get("rocAuc")))
        metric_or_na("Temporal churn ROC-AUC", f"{float(val):.3f}" if isinstance(val, (int, float)) else None,
                     "Target is 0.88; baseline is below target.")
    with metric_cols[2]:
        metric_or_na("Inventory products", f"{inventory['ProductID'].nunique():,}" if inventory is not None and "ProductID" in inventory else None,
                     "Historical demand-based estimates, not verified stock positions.")


# ---------- Page 2: General overview (project introduction) ----------
elif page == "General Overview":
    st.markdown(
        """
        <div class="hero">
          <h1>Welcome to RetailPulse</h1>
          <p>AI-powered retail analytics for understanding sales, customers, churn risk, and product demand.</p>
          <p class="small-note">This project turns historical Online Retail transactions into practical insights and decision-support estimates.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("What does RetailPulse do?")
    st.write(
        "RetailPulse brings sales analysis, customer analytics, machine-learning forecasts, "
        "and inventory planning into one dashboard. It helps explore what has sold, which "
        "customer groups behave differently, which customers may be at risk of becoming inactive, "
        "and which products may need replenishment review."
    )

    st.subheader("Interactive project explorer")
    topic_details = {
        "Sales & Forecasting": (
            "Explore historical sales and compare saved Prophet, LSTM, and ensemble holdout forecasts.",
            "These are historical evaluation results, not live predictions. Prophet is the strongest current baseline, but its recorded non-zero-day MAPE remains above the project target."
        ),
        "Customer Segmentation": (
            "Explore RFM behaviour, K-Means groups, DBSCAN outliers, and PCA visualisations.",
            "PCA provides a two-dimensional projection for exploration. Segment IDs are algorithmic labels; interpret them using RFM summaries."
        ),
        "Churn Risk": (
            "Review customer risk scores, ranked customers, and model feature explanations.",
            "Risk scores are prioritisation signals, not proof that a customer will churn. The current temporal ROC-AUC remains below the project target."
        ),
        "Inventory Planning": (
            "Review estimated demand, reorder guidance, and products flagged for manual review.",
            "These are historical demand-based estimates. Stock-on-hand and unit-cost data are unavailable, so current inventory and realised savings cannot be verified."
        ),
        "Data Quality & Exports": (
            "Inspect missing values, validation findings, and downloadable analysis artifacts.",
            "Check data-quality limitations before interpreting results. The cleaned transaction export is large and may take time to prepare."
        ),
        "Monitoring & Model Health": (
            "Inspect saved monitoring metrics and data-drift diagnostic reports.",
            "Distribution drift is a reason to investigate, not proof of model degradation. The saved churn-score comparison used a row-order split, not a chronological split."
        ),
    }
    selected_topic = st.selectbox(
        "Choose an analytics area",
        list(topic_details.keys()),
        key="generalOverviewTopic",
        help="Choose an area to see what it offers and how to interpret its results.",
    )
    summary_text, detail_text = topic_details[selected_topic]
    st.markdown(f"**{summary_text}**")
    st.info(detail_text)

    workflow_tab, methods_tab, interpretation_tab = st.tabs(
        ["Workflow", "Methods", "Interpretation guide"]
    )
    with workflow_tab:
        st.write(
            "Follow the project from data cleaning and validation through sales analysis, "
            "customer segmentation, churn-risk modelling, inventory review, and monitoring."
        )
    with methods_tab:
        st.write(
            "Methods include exploratory data analysis, RFM features, K-Means, DBSCAN, PCA, "
            "Prophet, LSTM, XGBoost, SHAP explanations, historical demand estimates, and drift diagnostics."
        )
    with interpretation_tab:
        st.write(
            "Treat model outputs as decision-support evidence, not guarantees. Forecast and churn "
            "targets have not all been met, and inventory impact has not been verified against "
            "actual stock levels or measured business outcomes."
        )

    st.subheader("Explore the project")
    cards = st.columns(2)
    with cards[0]:
        st.markdown("**Sales & Forecasting**")
        st.write("Explore historical sales, compare Prophet and LSTM forecasts, and review holdout performance.")
        st.markdown("**Customer Segmentation**")
        st.write("Understand customer groups using RFM behaviour, K-Means, DBSCAN, and PCA visualisations.")
        st.markdown("**Churn Risk**")
        st.write("Review modelled inactivity risk, important predictive features, and SHAP explanations.")
    with cards[1]:
        st.markdown("**Inventory Planning**")
        st.write("View estimated product demand, reorder guidance, and products that need manual review.")
        st.markdown("**Data Quality & Exports**")
        st.write("Inspect data-quality findings and export available reports or data for further analysis.")
        st.markdown("**Executive Overview**")
        st.write("See a quick snapshot of dataset scale, sales history, customer segments, and model metrics.")

    st.subheader("Dataset at a glance")
    if cleaned is None:
        show_missing_file("data/processedData/cleanedTransactions.csv")
    else:
        date_col = find_column(cleaned, ["invoiceDate", "InvoiceDate"])
        customer_col = find_column(cleaned, ["customerId", "Customer ID"])
        product_col = find_column(cleaned, ["ProductID"])
        dates = pd.to_datetime(cleaned[date_col], errors="coerce") if date_col else pd.Series(dtype="datetime64[ns]")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Transaction rows", f"{len(cleaned):,}")
        m2.metric("Dataset fields", f"{len(cleaned.columns):,}")
        m3.metric("Distinct products", f"{cleaned[product_col].nunique():,}" if product_col else "N/A")
        m4.metric("Identified customers", f"{cleaned[customer_col].nunique(dropna=True):,}" if customer_col else "N/A")
        if not dates.dropna().empty:
            st.caption(f"Historical period: {dates.min():%d %b %Y} – {dates.max():%d %b %Y}")

    st.subheader("Important to know")
    st.write(
        "This is a historical analytics demonstration, not a live retail or inventory system. "
        "Some customer IDs are missing, actual stock-on-hand and unit-cost values are unavailable, "
        "and the current forecasting and churn models have not met every target. Inventory values "
        "are estimates for review—not verified stock positions or purchase orders."
    )
    st.caption("Choose a section from the sidebar to explore the analysis. Detailed field-level information is available under Data Quality & Exports.")


# ---------- Page 3: Forecasting ----------
elif page == "Sales & Forecasting":
    st.title("Sales & Forecasting")
    st.write("Explore historical sales and compare saved 30-day holdout predictions against a 7-day seasonal-naive baseline.")

    if daily_sales is not None:
        date_col = find_column(daily_sales, ["date", "ds"])
        sales_col = find_column(daily_sales, ["dailySales", "y", "sales"])
        if date_col and sales_col:
            ts = daily_sales.copy()
            ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
            ts[sales_col] = pd.to_numeric(ts[sales_col], errors="coerce")
            ts = ts.dropna(subset=[date_col, sales_col]).sort_values(date_col)
            start, end = st.columns(2)
            min_date = ts[date_col].min().date()
            max_date = ts[date_col].max().date()
            with start:
                date_from = st.date_input("Start date", value=min_date, min_value=min_date, max_value=max_date)
            with end:
                date_to = st.date_input("End date", value=max_date, min_value=min_date, max_value=max_date)
            if date_from > date_to:
                st.error("Start date must be on or before end date.")
            else:
                filtered_ts = ts.loc[
                    (ts[date_col].dt.date >= date_from) & (ts[date_col].dt.date <= date_to)
                ]
                st.plotly_chart(
                    px.line(filtered_ts, x=date_col, y=sales_col, title="Daily sales over selected period",
                            labels={date_col: "Date", sales_col: "Sales"}),
                    use_container_width=True,
                )
                st.metric("Sales in selected period", f"{filtered_ts[sales_col].sum():,.2f}")
                st.metric("Average daily sales", f"{filtered_ts[sales_col].mean():,.2f}")
        else:
            show_missing_file("data/processedData/dailySales.csv", "Could not identify date and sales columns.")
    else:
        show_missing_file("data/processedData/dailySales.csv")

    st.subheader("Holdout model comparison")
    model_frames = []
    if prophet is not None:
        date_c = find_column(prophet, ["ds", "date"])
        actual_c = find_column(prophet, ["actualSales", "actual"])
        pred_c = find_column(prophet, ["forecastSales", "yhat", "prophetForecast"])
        if date_c and actual_c and pred_c:
            p = prophet[[date_c, actual_c, pred_c]].copy()
            p.columns = ["date", "actualSales", "prediction"]
            p["model"] = "Prophet"
            model_frames.append(p)
    if lstm is not None:
        date_c = find_column(lstm, ["date", "ds"])
        actual_c = find_column(lstm, ["actualSales", "actual"])
        pred_c = find_column(lstm, ["lstmForecast", "prediction", "forecast"])
        if date_c and actual_c and pred_c:
            p = lstm[[date_c, actual_c, pred_c]].copy()
            p.columns = ["date", "actualSales", "prediction"]
            p["model"] = "LSTM"
            model_frames.append(p)
    if ensemble is not None:
        date_c = find_column(ensemble, ["date", "ds"])
        actual_c = find_column(ensemble, ["actualSales", "actual"])
        for pred_name, model_name in [("prophetForecast", "Prophet (ensemble file)"),
                                      ("lstmForecast", "LSTM (ensemble file)"),
                                      ("ensembleForecast", "Equal-weight ensemble")]:
            pred_c = find_column(ensemble, [pred_name])
            if date_c and actual_c and pred_c:
                p = ensemble[[date_c, actual_c, pred_c]].copy()
                p.columns = ["date", "actualSales", "prediction"]
                p["model"] = model_name
                model_frames.append(p)

    # Add a 7-day seasonal-naive baseline from the historical daily sales series.
    # Each holdout-day prediction uses sales from exactly seven days earlier.
    if daily_sales is not None:
        naive_date_c = find_column(daily_sales, ["date", "ds"])
        naive_sales_c = find_column(daily_sales, ["dailySales", "y", "sales"])
        if naive_date_c and naive_sales_c:
            naive = daily_sales[[naive_date_c, naive_sales_c]].copy()
            naive.columns = ["date", "prediction"]
            naive["date"] = pd.to_datetime(naive["date"], errors="coerce")
            naive["prediction"] = pd.to_numeric(naive["prediction"], errors="coerce")
            naive = naive.dropna(subset=["date", "prediction"]).sort_values("date")
            naive = naive.drop_duplicates("date", keep="last").set_index("date")
            naive["actualSales"] = naive["prediction"]
            naive["prediction"] = naive["prediction"].shift(7)
            naive = naive.dropna(subset=["prediction"]).reset_index()
            naive["model"] = "Seasonal naive (7-day)"
            # Restrict to dates represented by the saved model holdout forecasts.
            if model_frames:
                existing_dates = pd.to_datetime(
                    pd.concat(model_frames, ignore_index=True)["date"], errors="coerce"
                ).dropna().dt.normalize().unique()
                naive = naive[naive["date"].dt.normalize().isin(existing_dates)]
            if not naive.empty:
                model_frames.append(naive[["date", "actualSales", "prediction", "model"]])

    if model_frames:
        all_forecasts = pd.concat(model_frames, ignore_index=True)
        all_forecasts["date"] = pd.to_datetime(all_forecasts["date"], errors="coerce")
        for col in ["actualSales", "prediction"]:
            all_forecasts[col] = pd.to_numeric(all_forecasts[col], errors="coerce")
        all_forecasts = all_forecasts.dropna(subset=["date", "actualSales", "prediction"])
        chosen_models = st.multiselect(
            "Models to compare",
            options=sorted(all_forecasts["model"].unique()),
            default=sorted(all_forecasts["model"].unique()),
        )
        plot_df = all_forecasts[all_forecasts["model"].isin(chosen_models)]
        actual = plot_df[["date", "actualSales"]].drop_duplicates("date")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=actual["date"], y=actual["actualSales"], mode="lines+markers", name="Actual"))
        for model_name in chosen_models:
            model_df = plot_df[plot_df["model"] == model_name].sort_values("date")
            fig.add_trace(go.Scatter(x=model_df["date"], y=model_df["prediction"], mode="lines", name=model_name))
        fig.update_layout(title="30-day holdout: actual vs predictions", xaxis_title="Date", yaxis_title="Sales",
                          legend_title="Series")
        st.plotly_chart(fig, use_container_width=True)

        rows = []
        for model_name, group in plot_df.groupby("model"):
            error = group["actualSales"] - group["prediction"]
            nonzero = group["actualSales"] != 0
            rows.append({
                "Model": model_name,
                "MAE": float(error.abs().mean()),
                "RMSE": float(np.sqrt(np.mean(error ** 2))),
                "WAPE (%)": float(error.abs().sum() / group["actualSales"].abs().sum() * 100)
                if group["actualSales"].abs().sum() else np.nan,
                "MAPE on nonzero actual days (%)": float(
                    (error[nonzero].abs() / group.loc[nonzero, "actualSales"].abs()).mean() * 100
                ) if nonzero.any() else np.nan,
            })
        metric_df = pd.DataFrame(rows).sort_values("WAPE (%)")
        st.dataframe(metric_df, use_container_width=True, hide_index=True)
        download_csv_button(metric_df, "retailPulseForecastComparison.csv", "Download comparison metrics")
        st.caption("Metrics are recomputed from the holdout series. The seasonal-naive prediction for each day is the observed sales value from seven days earlier. MAPE excludes zero-actual days; WAPE includes all rows.")

        st.subheader("What-if scenario explorer")
        st.caption(
            "This applies a user-selected percentage adjustment to saved holdout predictions for scenario exploration. "
            "It does not retrain a model and is not a new future forecast."
        )
        scenario_change = st.slider(
            "Assumed sales change (%)",
            min_value=-30,
            max_value=30,
            value=0,
            step=5,
            help="Explore how a simple proportional change would affect the saved holdout predictions.",
        )
        if not plot_df.empty:
            scenario_df = plot_df.copy()
            scenario_df["scenarioPrediction"] = scenario_df["prediction"] * (1 + scenario_change / 100)
            scenario_fig = go.Figure()
            actual_scenario = scenario_df[["date", "actualSales"]].drop_duplicates("date").sort_values("date")
            scenario_fig.add_trace(go.Scatter(
                x=actual_scenario["date"], y=actual_scenario["actualSales"],
                mode="lines+markers", name="Actual sales",
            ))
            for model_name in chosen_models:
                model_scenario = scenario_df[scenario_df["model"] == model_name].sort_values("date")
                scenario_fig.add_trace(go.Scatter(
                    x=model_scenario["date"], y=model_scenario["scenarioPrediction"],
                    mode="lines", name=f"{model_name} — scenario",
                ))
            scenario_fig.update_layout(
                title=f"Illustrative holdout scenario ({scenario_change:+d}% applied)",
                xaxis_title="Date", yaxis_title="Sales", legend_title="Series",
            )
            st.plotly_chart(scenario_fig, use_container_width=True)
            scenario_cols = st.columns(3)
            baseline_total = float(plot_df["prediction"].sum()) if not plot_df.empty else 0.0
            scenario_total = float(scenario_df["scenarioPrediction"].sum())
            scenario_cols[0].metric("Baseline predicted total", f"{baseline_total:,.2f}")
            scenario_cols[1].metric("Scenario predicted total", f"{scenario_total:,.2f}")
            scenario_cols[2].metric("Difference", f"{scenario_total - baseline_total:+,.2f}")
            download_csv_button(
                scenario_df[["date", "model", "actualSales", "prediction", "scenarioPrediction"]],
                "retailPulseForecastWhatIfScenario.csv",
                "Download what-if scenario",
            )
        else:
            st.info("Select at least one model to explore a what-if scenario.")
    else:
        show_missing_file("data/processedData/*HoldoutForecast.csv", "No recognized holdout forecast files were found.")

    with st.expander("Forecasting methodology and limitations"):
        st.markdown(
            """
            - The currently available forecast is **aggregate daily sales**, not SKU-level demand.
            - Historical holdout results show Prophet performs better than the LSTM and equal-weight ensemble.
            - The current Prophet nonzero-day MAPE is approximately 27.5%, above the project target of 12%.
            - The history ends on 9 December 2011; this is not a live forecast.
            - A future operational version should use rolling-origin validation and compare against seasonal-naive baselines.
            """
        )


# ---------- Page 3: Segmentation ----------
elif page == "Customer Segmentation":
    st.title("Customer Segmentation")
    st.write("Explore customer segments, PCA projections, and segment-level RFM summaries.")

    if segments is None:
        show_missing_file("data/processedData/customerSegments.csv")
    else:
        cluster_col = find_column(segments, ["cluster", "kmeansCluster", "segment", "clusterLabel"])
        customer_col = find_column(segments, ["customerId", "customer_id", "Customer ID"])
        pc1_col = find_column(segments, ["PC1", "pca1", "principalComponent1"])
        pc2_col = find_column(segments, ["PC2", "pca2", "principalComponent2"])
        recency_col = find_column(segments, ["recencyDays", "recency", "Recency"])
        frequency_col = find_column(segments, ["frequency", "Frequency"])
        monetary_col = find_column(segments, ["monetary", "monetaryTotal", "Monetary"])

        filtered = segments.copy()
        if cluster_col:
            cluster_values = sorted(filtered[cluster_col].dropna().astype(str).unique())
            selected = st.multiselect("K-Means segment(s)", cluster_values, default=cluster_values)
            filtered = filtered[filtered[cluster_col].astype(str).isin(selected)]

        a, b, c = st.columns(3)
        with a:
            st.metric("Customers in selection", f"{len(filtered):,}")
        with b:
            st.metric("Segments represented", f"{filtered[cluster_col].nunique()}" if cluster_col else "N/A")
        with c:
            st.metric("RFM columns available", str(sum(x is not None for x in [recency_col, frequency_col, monetary_col])) + "/3")

        if pc1_col and pc2_col and cluster_col:
            scatter_df = filtered.copy()
            scatter_df[pc1_col] = pd.to_numeric(scatter_df[pc1_col], errors="coerce")
            scatter_df[pc2_col] = pd.to_numeric(scatter_df[pc2_col], errors="coerce")
            hover_cols = [c for c in [customer_col, recency_col, frequency_col, monetary_col] if c]
            scatter_df["_segmentLabel"] = scatter_df[cluster_col].astype(str)
            fig = px.scatter(
                scatter_df.dropna(subset=[pc1_col, pc2_col]),
                x=pc1_col, y=pc2_col, color="_segmentLabel",
                hover_data=hover_cols, title="PCA projection of customer segments",
                labels={pc1_col: "Principal component 1", pc2_col: "Principal component 2", "_segmentLabel": "Segment"},
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("PCA is a 2D projection of the engineered RFM feature space; visual separation does not alone prove business value.")
        else:
            st.warning("PCA columns or a recognized cluster column were not found in customerSegments.csv.")
            st.write("Available columns:", list(segments.columns))

        if cluster_col and any([recency_col, frequency_col, monetary_col]):
            agg_map = {}
            if customer_col:
                agg_map[customer_col] = "nunique"
            if recency_col:
                agg_map[recency_col] = "median"
            if frequency_col:
                agg_map[frequency_col] = "median"
            if monetary_col:
                agg_map[monetary_col] = "median"
            if agg_map:
                summary = filtered.groupby(cluster_col, dropna=False).agg(agg_map).reset_index()
                summary = summary.rename(columns={customer_col: "Unique customers"} if customer_col else {})
                st.subheader("Segment profile")
                st.dataframe(summary, use_container_width=True, hide_index=True)
                download_csv_button(summary, "retailPulseSegmentProfiles.csv")
        st.subheader("Customer-level segment assignments")
        st.dataframe(filtered.head(1000), use_container_width=True, hide_index=True)
        download_csv_button(filtered, "retailPulseCustomerSegments.csv", "Download filtered segment assignments")

        with st.expander("Segmentation interpretation"):
            st.markdown(
                """
                - K-Means k=6 was selected for business interpretability, although k=2 had the highest silhouette score in the recorded sweep.
                - Segment labels are algorithmic IDs, not inherent customer types; interpretations should be grounded in the RFM summaries.
                - DBSCAN noise points are outliers under the chosen density settings, not automatically errors or low-value customers.
                """
            )
        if segmentation_report_path.exists():
            with st.expander("Saved segmentation report"):
                st.text(read_text(segmentation_report_path))


# ---------- Page 4: Churn ----------
elif page == "Churn Risk":
    st.title("Customer Churn Risk")
    st.write("Explore model scores and identify customers ranked highest by the saved churn model.")

    if churn is None:
        show_missing_file("reports/modelReports/churnTemporalRiskPredictions.csv")
    else:
        customer_col = find_column(churn, ["customerId", "customer_id", "Customer ID"])
        prob_col = find_column(churn, ["churnProbability", "probability", "churnProb", "riskScore", "predictedProbability"])
        pred_col = find_column(churn, ["predictedChurn", "churnPrediction", "prediction", "churn"])
        actual_col = find_column(churn, ["churned", "actualChurn", "churnLabel", "target", "isChurned"])
        risk_col = find_column(churn, ["riskBand", "riskCategory", "riskLevel", "churnRisk"])
        st.caption(f"Loaded `{churn_path.name}`.")
        if prob_col:
            churn[prob_col] = pd.to_numeric(churn[prob_col], errors="coerce")
            threshold = st.slider("High-risk score threshold", 0.0, 1.0, 0.5, 0.05)
            ranked = churn.sort_values(prob_col, ascending=False).copy()
            ranked["dashboardRiskBand"] = np.where(ranked[prob_col] >= threshold, "Above threshold", "Below threshold")
            x1, x2, x3 = st.columns(3)
            x1.metric("Scored customers", f"{len(ranked):,}")
            x2.metric("Above threshold", f"{(ranked[prob_col] >= threshold).sum():,}")
            x3.metric("Mean predicted risk", f"{ranked[prob_col].mean():.1%}")
            fig = px.histogram(ranked, x=prob_col, nbins=30, color="dashboardRiskBand",
                               title="Distribution of churn scores",
                               labels={prob_col: "Predicted churn score"})
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Highest-risk customers")
            display_cols = [c for c in [customer_col, prob_col, risk_col, pred_col, actual_col] if c]
            if not display_cols:
                display_cols = ranked.columns.tolist()
            st.dataframe(ranked[display_cols].head(100), use_container_width=True, hide_index=True)
            download_csv_button(ranked, "retailPulseRankedChurnRisk.csv", "Download ranked churn scores")

            # Risk-band audit: compare predicted probabilities with observed labels.
            if risk_col and actual_col:
                st.subheader("Risk-band performance audit")
                audit = churn[[risk_col, actual_col, prob_col]].copy()
                audit[actual_col] = pd.to_numeric(audit[actual_col], errors="coerce")
                audit[prob_col] = pd.to_numeric(audit[prob_col], errors="coerce")
                audit = audit.dropna(subset=[risk_col, actual_col, prob_col])
                audit = audit[audit[actual_col].isin([0, 1])]
                if not audit.empty:
                    band_audit = audit.groupby(risk_col, dropna=False).agg(
                        customers=(actual_col, "size"),
                        observed_churn_rate=(actual_col, "mean"),
                        avg_predicted_probability=(prob_col, "mean"),
                    ).reset_index()
                    band_audit["observed_churn_rate"] *= 100
                    band_audit["avg_predicted_probability"] *= 100
                    band_audit = band_audit.rename(columns={
                        risk_col: "Risk band",
                        "customers": "Customers",
                        "observed_churn_rate": "Observed churn (%)",
                        "avg_predicted_probability": "Mean predicted probability (%)",
                    })
                    st.dataframe(
                        band_audit.style.format({
                            "Observed churn (%)": "{:.2f}%",
                            "Mean predicted probability (%)": "{:.2f}%",
                        }),
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.plotly_chart(
                        px.bar(
                            band_audit.melt(
                                id_vars=["Risk band"],
                                value_vars=["Observed churn (%)", "Mean predicted probability (%)"],
                                var_name="Measure", value_name="Percent",
                            ),
                            x="Risk band", y="Percent", color="Measure", barmode="group",
                            title="Observed churn vs predicted probability by risk band",
                            labels={"Percent": "Percent (%)"},
                        ),
                        use_container_width=True,
                    )
                    st.caption(
                        "Descriptive audit on the saved labeled predictions. Ordered observed rates support risk ranking, "
                        "but group averages alone do not establish probability calibration or future performance."
                    )
                    download_csv_button(band_audit, "retailPulseChurnRiskBandAudit.csv", "Download risk-band audit")
                else:
                    st.info("Risk-band audit needs valid binary churn labels and numeric probabilities.")
            else:
                st.info("Risk-band audit unavailable: this file must include a risk-band column and observed churn labels.")
        else:
            st.warning("A probability/score column was not recognized. Available columns:")
            st.write(churn.columns.tolist())
            st.dataframe(churn.head(100), use_container_width=True, hide_index=True)
            download_csv_button(churn, "retailPulseChurnPredictions.csv")

        if churn_metrics:
            st.subheader("Temporal validation metrics")
            st.json(churn_metrics)
        with st.expander("Model interpretation and caveats"):
            st.markdown(
                """
                - The temporal baseline recorded ROC-AUC 0.7633 and precision@top 20% 0.7381, both below their respective targets.
                - Scores are prioritization signals, not proof that a customer will churn.
                - Historical cutoffs are based on a dataset ending in 2011; validate on recent data before operational use.
                - Use SHAP explanations and customer context before selecting retention actions.
                """
            )
        st.subheader("Model explainability: feature importance / SHAP")
        if churn_shap is not None and not churn_shap.empty:
            st.caption(f"Loaded explanation artifact: `{churn_shap_path.name}`")
            importance_col = find_column(churn_shap, [
                "meanAbsShap", "mean_abs_shap", "meanAbsSHAP", "importance",
                "importanceScore", "gain", "weight", "shapImportance"
            ])
            feature_col = find_column(churn_shap, ["feature", "featureName", "Feature", "name"])
            if importance_col:
                churn_shap[importance_col] = pd.to_numeric(churn_shap[importance_col], errors="coerce")
                plot_importance = churn_shap.dropna(subset=[importance_col]).nlargest(15, importance_col)
                if feature_col and not plot_importance.empty:
                    st.plotly_chart(
                        px.bar(plot_importance.sort_values(importance_col), x=importance_col, y=feature_col,
                               orientation="h", title="Top churn model drivers",
                               labels={importance_col: "Importance", feature_col: "Feature"}),
                        use_container_width=True,
                    )
                st.dataframe(churn_shap, use_container_width=True, hide_index=True)
            else:
                st.info("The importance file is present, but its score column was not recognized. Showing the saved data below.")
                st.dataframe(churn_shap, use_container_width=True, hide_index=True)
            download_csv_button(churn_shap, "retailPulseChurnFeatureImportance.csv", "Download feature importance")
        else:
            st.warning("No recognized SHAP/feature-importance CSV was found. Expected one of: churnShapImportance.csv, churnTemporalFeatureImportance.csv, churnFeatureImportance.csv.")

        shap_image = first_existing(FIGURE_DIR / "churnShapSummary.png", FIGURE_DIR / "churnFeatureImportance.png",
                                    FIGURE_DIR / "churnTemporalFeatureImportance.png")
        if shap_image:
            with st.expander("Saved model explanation image"):
                st.image(str(shap_image), caption=shap_image.name, use_container_width=True)

        if churn_report_path.exists():
            with st.expander("Saved temporal validation report"):
                st.text(read_text(churn_report_path))


# ---------- Page 5: Inventory ----------
elif page == "Inventory Planning":
    st.title("Inventory Planning")
    st.write("Historical product-level demand estimates, safety stock, and reorder-point scenarios.")

    if inventory is None:
        show_missing_file("reports/modelReports/inventoryOptimizationRecommendations.csv")
    else:
        demand_col = find_column(inventory, ["estimatedDemandNext30Days"])
        status_col = find_column(inventory, ["recommendationStatus"])
        product_col = find_column(inventory, ["ProductID"])
        daily_col = find_column(inventory, ["estimatedDailyDemand"])
        reorder_col = find_column(inventory, ["estimatedReorderPointUnits"])
        safety_col = find_column(inventory, ["estimatedSafetyStockUnits"])
        target_col = find_column(inventory, ["estimatedTargetStockUnits"])
        recent_col = find_column(inventory, ["unitsSoldLast28Days"])

        view = inventory.copy()
        if status_col:
            statuses = sorted(view[status_col].dropna().astype(str).unique())
            selected_statuses = st.multiselect("Recommendation category", statuses, default=statuses)
            view = view[view[status_col].astype(str).isin(selected_statuses)]
        if demand_col:
            view[demand_col] = pd.to_numeric(view[demand_col], errors="coerce")
            max_demand = float(view[demand_col].max()) if view[demand_col].notna().any() else 0.0
            if max_demand > 0:
                max_filter = st.slider("Maximum estimated 30-day demand to display", 0.0, max_demand,
                                       max_demand, max(max_demand / 100, 0.1))
                view = view[view[demand_col] <= max_filter]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Products in view", f"{view[product_col].nunique():,}" if product_col else f"{len(view):,}")
        k2.metric("Estimated 30-day demand", f"{view[demand_col].sum():,.0f}" if demand_col else "N/A")
        k3.metric("Median reorder point", f"{view[reorder_col].median():,.1f}" if reorder_col else "N/A")
        k4.metric("Median safety stock", f"{view[safety_col].median():,.1f}" if safety_col else "N/A")

        if product_col and demand_col:
            top_n = st.slider("Number of products in demand chart", 5, 30, 15)
            top = view.nlargest(top_n, demand_col).sort_values(demand_col)
            st.plotly_chart(
                px.bar(top, x=demand_col, y=product_col, orientation="h",
                       title=f"Top {len(top)} products by estimated next-30-day demand",
                       labels={demand_col: "Estimated units", product_col: "Product ID"}),
                use_container_width=True,
            )
        show_cols = [c for c in [
            product_col, "lastSaleDate", "daysSinceLastSale", recent_col, daily_col,
            demand_col, "averageDailyDemandLast90", safety_col, reorder_col, target_col,
            "estimatedReplenishmentUnitsIfStockZero", status_col
        ] if c and c in view.columns]
        st.subheader("Product-level recommendations")
        st.dataframe(view[show_cols].sort_values(demand_col, ascending=False) if demand_col else view[show_cols],
                     use_container_width=True, hide_index=True)
        download_csv_button(view, "retailPulseInventoryRecommendations.csv", "Download filtered inventory recommendations")

        st.subheader("Demand confidence and manual-review flags")
        if inventory_confidence is not None and not inventory_confidence.empty:
            st.caption(f"Loaded confidence audit: `{inventory_confidence_path.name}`")
            flag_col = find_column(inventory_confidence, [
                "manualReview", "manualReviewFlag", "needsManualReview", "reviewFlag",
                "confidenceFlag", "flagForReview", "requiresReview", "reviewRequired"
            ])
            reason_col = find_column(inventory_confidence, [
                "reviewReason", "manualReviewReason", "confidenceReason", "reason", "flags", "auditReason"
            ])
            if flag_col:
                flag_values = inventory_confidence[flag_col].astype(str).str.strip().str.lower()
                flagged_mask = flag_values.isin(["true", "1", "yes", "y", "manual review", "review"])
                if not flagged_mask.any():
                    flagged_mask = ~flag_values.isin(["false", "0", "no", "n", "none", "nan", "", "ok", "pass"])
                c1, c2 = st.columns(2)
                c1.metric("Products flagged for manual review", f"{int(flagged_mask.sum()):,}")
                c2.metric("Products assessed", f"{len(inventory_confidence):,}")
                if reason_col:
                    reasons = inventory_confidence.loc[flagged_mask, reason_col].fillna("Unspecified").astype(str).value_counts().head(10)
                    if not reasons.empty:
                        st.plotly_chart(px.bar(reasons.rename_axis("Review reason").reset_index(name="Products"),
                                               x="Products", y="Review reason", orientation="h",
                                               title="Most common manual-review reasons"), use_container_width=True)
                review_df = inventory_confidence.loc[flagged_mask].copy()
                st.dataframe(review_df.head(500), use_container_width=True, hide_index=True)
                download_csv_button(review_df, "retailPulseInventoryManualReview.csv", "Download flagged products")
                st.caption("The confidence audit flags unusual demand patterns for human review; it does not prove that a transaction is erroneous.")
            else:
                st.warning("Confidence audit exists, but a review-flag column could not be recognized. Showing the audit so the column names can be checked.")
                st.dataframe(inventory_confidence.head(500), use_container_width=True, hide_index=True)
                download_csv_button(inventory_confidence, "retailPulseInventoryDemandConfidenceAudit.csv")
        else:
            st.warning("Demand-confidence audit not found. Expected `reports/modelReports/inventoryDemandConfidenceAudit.csv`. Run `auditInventoryDemandConfidence.py` from the project root, then refresh the dashboard.")

        st.warning(
            "Stock-on-hand and unit-cost data are missing. These are historical planning estimates; "
            "they are not purchase orders and do not establish actual savings or stockout reduction."
        )
        with st.expander("Assumptions used"):
            st.markdown(
                """
                - Supplier lead time: 14 days (assumed).
                - Review period: 7 days (assumed).
                - Approximate service level: 95%.
                - Daily demand combines the recent 28-day rate (60% weight) and 90-day rate (40% weight).
                - Historical zero-sale days are included in the demand windows.
                - Dataset ends on 9 December 2011.
                """
            )
        if inventory_report_path.exists():
            with st.expander("Inventory demand audit report"):
                st.text(read_text(inventory_report_path))


# ---------- Page 6: Data quality & exports ----------
elif page == "Data Quality & Exports":
    st.title("Data Quality & Exports")
    st.write("Inspect data availability and download the current analysis outputs.")

    st.subheader("Download analysis bundle")
    st.write(
        "Package available forecasts, segmentation, churn, inventory, data-quality, and monitoring artifacts "
        "into one ZIP for review or submission. The large cleaned transaction CSV is excluded from this bundle."
    )
    bundle_bytes, bundle_files = build_analysis_export_bundle(ROOT)
    b1, b2 = st.columns(2)
    b1.metric("Available files included", f"{len(bundle_files):,}")
    b2.metric("Bundle size", f"{len(bundle_bytes) / (1024 * 1024):.2f} MB")
    st.download_button(
        "Download RetailPulse analysis bundle (ZIP)",
        data=bundle_bytes,
        file_name="retailPulse_analysis_bundle.zip",
        mime="application/zip",
        help="Includes available saved analysis artifacts and a README with interpretation notes.",
    )
    with st.expander("View files included in the bundle"):
        if bundle_files:
            st.code("\n".join(bundle_files), language="text")
        else:
            st.info("No saved analysis artifacts were found; the ZIP contains only its explanatory README and file list.")

    if cleaned is not None:
        st.subheader("Cleaned transaction dataset")
        date_col = find_column(cleaned, ["invoiceDate"])
        product_col = find_column(cleaned, ["ProductID"])
        customer_col = find_column(cleaned, ["customerId"])
        quantity_col = find_column(cleaned, ["quantity"])
        sales_col = find_column(cleaned, ["SalesAmount"])
        cols = st.columns(4)
        cols[0].metric("Rows", f"{len(cleaned):,}")
        cols[1].metric("Columns", f"{len(cleaned.columns)}")
        cols[2].metric("Unique products", f"{cleaned[product_col].nunique():,}" if product_col else "N/A")
        cols[3].metric("Missing customer IDs", f"{cleaned[customer_col].isna().sum():,}" if customer_col else "N/A")

        missing_df = cleaned.isna().sum().rename("missingRows").reset_index().rename(columns={"index": "column"})
        missing_df["missingPercent"] = (missing_df["missingRows"] / max(len(cleaned), 1) * 100).round(2)
        missing_df = missing_df.sort_values("missingRows", ascending=False)
        st.plotly_chart(px.bar(missing_df.head(15).sort_values("missingRows"),
                               x="missingRows", y="column", orientation="h",
                               title="Columns with the most missing values"),
                        use_container_width=True)
        st.dataframe(missing_df, use_container_width=True, hide_index=True)
        # Download the full cleaned data only on explicit user click to avoid accidental large transfers.
        with st.expander("Download cleaned dataset (large file)"):
            st.caption("The CSV is approximately 208 MB; download may take time.")
            if st.button("Prepare cleaned transaction CSV"):
                st.session_state["prepareCleanedDownload"] = True
            if st.session_state.get("prepareCleanedDownload"):
                st.download_button(
                    "Download cleanedTransactions.csv",
                    data=cleaned.to_csv(index=False).encode("utf-8"),
                    file_name="cleanedTransactions.csv",
                    mime="text/csv",
                )
    else:
        show_missing_file("data/processedData/cleanedTransactions.csv")

    st.subheader("Available reports")
    artifacts = [
        ("EDA report", REPORT_DIR / "edaReport.txt"),
        ("Great Expectations validation", REPORT_DIR / "greatExpectationsValidation.txt"),
        ("Segmentation report", segmentation_report_path),
        ("Churn temporal validation", churn_report_path),
        ("Inventory demand audit", inventory_report_path),
        ("Forecast comparison", REPORT_DIR / "forecastModelComparison.csv"),
        ("Forecast ensemble metrics", REPORT_DIR / "forecastEnsembleMetrics.json"),
        ("Churn temporal metrics", REPORT_DIR / "churnTemporalValidationMetrics.json"),
        ("Inventory summary", REPORT_DIR / "inventoryOptimizationSummary.json"),
    ]
    for label, path in artifacts:
        if path.exists():
            with st.expander(f"{label} — {path.name}"):
                if path.suffix.lower() == ".csv":
                    artifact_df = read_csv_if_exists(str(path))
                    if artifact_df is not None:
                        st.dataframe(artifact_df, use_container_width=True, hide_index=True)
                        download_csv_button(artifact_df, path.name)
                elif path.suffix.lower() == ".json":
                    obj = read_json_if_exists(str(path))
                    st.json(obj)
                    st.download_button("Download JSON", path.read_bytes(), file_name=path.name, mime="application/json")
                else:
                    st.text(read_text(path))
                    st.download_button("Download report", path.read_bytes(), file_name=path.name, mime="text/plain")
    st.caption("The Great Expectations report previously recorded 36 passing checks and 0 failures. Re-run validation after changing the data pipeline.")

# ---------- Monitoring & Model Health ----------
elif page == "Monitoring & Model Health":
    st.title("Monitoring & Model Health")
    st.write(
        "Review historical model performance, compare forecast errors, and inspect data-drift diagnostics. "
        "This page reports saved offline assessments; it is not a live production alerting system."
    )

    summary_path = REPORT_DIR / "modelMonitoringSummary.json"
    forecast_monitoring_path = REPORT_DIR / "forecastMonitoringMetrics.csv"
    churn_monitoring_path = REPORT_DIR / "churnMonitoringMetrics.json"
    summary = read_json_if_exists(str(summary_path)) or {}
    forecast_monitoring = read_csv_if_exists(str(forecast_monitoring_path))
        # Add the seasonal-naive baseline to the monitoring table.
    # This baseline is calculated on the same historical holdout.
    baseline_path = REPORT_DIR / "forecastBaselineComparison.json"
    baseline_data = read_json_if_exists(str(baseline_path)) or {}

    baseline_models = baseline_data.get("models", {})
    seasonal_naive = baseline_models.get("seasonalNaive7Day", {})

    if seasonal_naive:
        baseline_row = {
            "model": "Seasonal Naive (7-day)",
            "MAE": seasonal_naive.get("MAE"),
            "RMSE": seasonal_naive.get("RMSE"),
            "MAPE_nonzero_days_percent": seasonal_naive.get(
                "MAPE_nonzero_days_percent"
            ),
            "WAPE_percent": seasonal_naive.get("WAPE_percent"),
        }

        if forecast_monitoring is None:
            forecast_monitoring = pd.DataFrame([baseline_row])
        elif not forecast_monitoring.empty:
            if "model" not in forecast_monitoring.columns:
                forecast_monitoring = forecast_monitoring.rename(
                    columns={"Model": "model"}
                )

            if "model" in forecast_monitoring.columns:
                forecast_monitoring = forecast_monitoring[
                    forecast_monitoring["model"] != "Seasonal Naive (7-day)"
                ]
                forecast_monitoring = pd.concat(
                    [forecast_monitoring, pd.DataFrame([baseline_row])],
                    ignore_index=True,
                )
    churn_monitoring = read_json_if_exists(str(churn_monitoring_path)) or {}

    def display_num(value, suffix="", digits=2):
        try:
            number = float(value)
            return f"{number:,.{digits}f}{suffix}" if np.isfinite(number) else "N/A"
        except (TypeError, ValueError):
            return "N/A"

    # High-level monitoring state
    overall = summary.get("overall_status", "MONITORING_NOT_RUN")
    statuses = summary.get("statuses", {})
    status_color = "🟢" if overall == "COMPLETED_WITH_CAVEATS" else "🟠"
    st.subheader("Monitoring status")
    st.markdown(f"### {status_color} {overall.replace('_', ' ').title()}")
    generated_at = summary.get("generated_at_utc")
    if generated_at:
        st.caption(f"Last generated (UTC): {generated_at}")
    if not summary:
        st.info("Monitoring artifacts have not been generated yet. Run `python src/monitoring/monitorModels.py` from the project root, then refresh this page.")

    status_cols = st.columns(4)
    status_labels = [
        ("Forecast metrics", "forecast_metrics"),
        ("Churn metrics", "churn_metrics"),
        ("Sales data drift", "sales_data_drift"),
        ("Churn score diagnostic", "churn_score_distribution"),
    ]
    for col, (label, key) in zip(status_cols, status_labels):
        value = statuses.get(key, "NOT_AVAILABLE")
        col.metric(label, str(value).replace("_", " ").title())

    # Forecast monitoring: summarize before showing the full table
    st.divider()
    st.subheader("Forecast performance monitoring")
    if forecast_monitoring is not None and not forecast_monitoring.empty:
        fm = forecast_monitoring.copy()
        # Include the saved seasonal-naive baseline in the same holdout leaderboard.
        baseline_path = REPORT_DIR / "simpleForecastBaselineResults.json"
        baseline_payload = read_json_if_exists(str(baseline_path)) or {}
        baseline_metrics = baseline_payload.get("models", {}).get("seasonalNaive7Day", {})
        if baseline_metrics and not ("model" in fm.columns and fm["model"].astype(str).str.lower().str.contains("seasonal naive").any()):
            holdout_start = baseline_payload.get("holdoutStart")
            holdout_end = baseline_payload.get("holdoutEnd")
            baseline_row = {column: None for column in fm.columns}
            mapped_values = {
                "model": "Seasonal naive (7-day)",
                "observations": baseline_payload.get("holdoutDays", 30),
                "mae": baseline_metrics.get("MAE"),
                "rmse": baseline_metrics.get("RMSE"),
                "mape_nonzero_actual_percent": baseline_metrics.get("MAPE_nonzero_days_percent"),
                "wape_percent": baseline_metrics.get("WAPE_percent"),
                "mean_error_forecast_minus_actual": None,
                "status": "BASELINE",
                "source_file": str(baseline_path.relative_to(ROOT)).replace("\\", "/"),
                "date_start": holdout_start,
                "date_end": holdout_end,
            }
            for column, value in mapped_values.items():
                if column in baseline_row:
                    baseline_row[column] = value
            fm = pd.concat([fm, pd.DataFrame([baseline_row])], ignore_index=True)
        for metric_col in ["mae", "rmse", "mape_nonzero_actual_percent", "wape_percent", "observations"]:
            if metric_col in fm.columns:
                fm[metric_col] = pd.to_numeric(fm[metric_col], errors="coerce")
        valid_wape = fm.dropna(subset=["wape_percent"]) if "wape_percent" in fm.columns else pd.DataFrame()
        if not valid_wape.empty:
            best = valid_wape.loc[valid_wape["wape_percent"].idxmin()]
            best_name = str(best.get("model", "Unknown"))
            best_wape = display_num(best.get("wape_percent"), "%")
        else:
            best_name, best_wape = "N/A", "N/A"
        k1, k2, k3 = st.columns(3)
        k1.metric("Models with valid WAPE", f"{len(valid_wape)}")
        k2.metric("Lowest recorded WAPE", best_wape, help="Lower is better; this uses the saved holdout, not a new test set.")
        k3.metric("Best model on this holdout", best_name)

        chart_metrics = [c for c in ["wape_percent", "mape_nonzero_actual_percent"] if c in fm.columns]
        if "model" in fm.columns and chart_metrics:
            chart = fm.melt(id_vars="model", value_vars=chart_metrics, var_name="Metric", value_name="Error (%)")
            chart["Metric"] = chart["Metric"].map({
                "wape_percent": "WAPE (%)", "mape_nonzero_actual_percent": "Non-zero-day MAPE (%)"
            }).fillna(chart["Metric"])
            chart = chart.dropna(subset=["Error (%)"])
            if not chart.empty:
                st.plotly_chart(
                    px.bar(chart, x="model", y="Error (%)", color="Metric", barmode="group",
                           title="Forecast error comparison (lower is better)",
                           hover_data={"Error (%)": ":.2f"}),
                    use_container_width=True,
                )
        with st.expander("View forecast metric details"):
            st.dataframe(fm, use_container_width=True, hide_index=True)
        download_csv_button(fm, "retailPulseForecastMonitoringMetrics.csv")
        if best_name == "Prophet":
            st.info("The lowest recorded WAPE is selected from the saved model forecasts and the 7-day seasonal-naive baseline. This is a historical holdout comparison, not a guarantee of future performance.")
        else:
            st.caption("The best model label refers only to the saved holdout comparison and does not establish future production performance.")
    else:
        show_missing_file("reports/modelReports/forecastMonitoringMetrics.csv", "Forecast monitoring metrics are unavailable. Run the monitoring script first.")

    # Churn monitoring
    st.divider()
    st.subheader("Churn model monitoring")
    if churn_monitoring:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Evaluated rows", f"{int(churn_monitoring.get('observations', 0)):,}")
        c2.metric("ROC-AUC", display_num(churn_monitoring.get("roc_auc"), digits=3))
        c3.metric("Average precision", display_num(churn_monitoring.get("average_precision"), digits=3))
        c4.metric("Observed churn rate", display_num(churn_monitoring.get("observed_churn_rate_percent"), "%"))
        m1, m2, m3 = st.columns(3)
        m1.metric("Mean predicted risk", display_num(churn_monitoring.get("mean_predicted_churn_probability_percent"), "%"))
        m2.metric("Precision at 0.5", display_num(churn_monitoring.get("precision_at_0_5"), digits=3))
        m3.metric("Recall at 0.5", display_num(churn_monitoring.get("recall_at_0_5"), digits=3))
        st.caption(churn_monitoring.get("note", "Metrics are based on saved historical evaluation artifacts."))
        with st.expander("View churn metric JSON"):
            st.json(churn_monitoring)
        st.download_button("Download churn monitoring metrics", data=churn_monitoring_path.read_bytes(),
                           file_name=churn_monitoring_path.name, mime="application/json")
    else:
        show_missing_file("reports/modelReports/churnMonitoringMetrics.json", "Churn monitoring metrics are unavailable. Run the monitoring script first.")

    # Drift interpretation and saved diagnostics
    st.divider()
    st.subheader("Data drift diagnostics")
    sales_drift = summary.get("sales_data_drift", {}) if isinstance(summary, dict) else {}
    score_drift = summary.get("churn_score_distribution", {}) if isinstance(summary, dict) else {}
    drift_tabs = st.tabs(["Sales data drift", "Churn score distribution", "Monitoring report"])
    with drift_tabs[0]:
        st.write("Compares the earlier 70% of daily-sales history with the later 30%, ordered by date.")
        if sales_drift:
            d1, d2, d3 = st.columns(3)
            d1.metric("Diagnostic status", str(sales_drift.get("status", "UNKNOWN")).replace("_", " ").title())
            d2.metric("Reference rows", f"{int(sales_drift.get('reference_rows', 0)):,}")
            d3.metric("Current rows", f"{int(sales_drift.get('current_rows', 0)):,}")
            if sales_drift.get("date_reference_start") and sales_drift.get("date_current_end"):
                st.caption(f"Reference: {sales_drift.get('date_reference_start')} to {sales_drift.get('date_reference_end')} · Comparison: {sales_drift.get('date_current_start')} to {sales_drift.get('date_current_end')}")

            # Show quantitative metrics generated by monitorModels.py, with a fallback to Evidently's KS result.
            sales_dm = sales_drift.get("distribution_metrics", {})
            sales_metrics = sales_drift.get("evidently_result", {}).get("metrics", [])
            sales_ks_p = sales_dm.get("ks_p_value")
            if sales_ks_p is None:
                sales_ks_p = next((m.get("value") for m in sales_metrics
                                   if "ValueDrift" in str(m.get("metric_name", ""))
                                   and "K-S p_value" in str(m.get("metric_name", ""))), None)
            if sales_dm:
                st.markdown("**Quantitative drift metrics**")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("KS statistic", display_num(sales_dm.get("ks_statistic"), digits=3))
                m2.metric("KS p-value", f"{float(sales_ks_p):.3g}" if sales_ks_p is not None else "N/A")
                m3.metric("PSI", display_num(sales_dm.get("population_stability_index"), digits=3))
                m4.metric("Standardized mean shift", display_num(sales_dm.get("standardized_mean_shift"), digits=3))
                st.caption(f"Heuristic assessment: {str(sales_dm.get('status', 'UNKNOWN')).replace('_', ' ').title()}. PSI thresholds are investigation guidelines, not universal guarantees.")
            elif sales_ks_p is not None:
                try:
                    sales_ks_p = float(sales_ks_p)
                    st.metric("Daily-sales KS-test p-value", f"{sales_ks_p:.3g}")
                except (TypeError, ValueError):
                    st.caption("A KS-test result exists, but its value could not be parsed.")
                st.info("KS/PSI/mean-shift metrics are not present in this summary yet. Rerun the updated monitoring script to populate them.")
            else:
                st.info("The saved report does not include readable quantitative drift metrics. Rerun the monitoring script after confirming the updated monitoring code is installed.")
            if sales_ks_p is not None:
                try:
                    if float(sales_ks_p) < 0.05:
                        st.warning("The daily-sales distribution differs significantly between the historical reference and comparison periods (KS p < 0.05). This is a drift signal, not proof of model degradation.")
                    else:
                        st.info("The KS test did not detect a statistically significant distribution difference at the 0.05 threshold. This does not prove the distributions are identical.")
                except (TypeError, ValueError):
                    pass
            if sales_drift.get("error"):
                st.error(sales_drift["error"])
        report_path = REPORT_DIR / "salesDataDriftReport.html"
        if report_path.exists():
            st.caption(f"Evidently report available ({report_path.stat().st_size:,} bytes). Download to inspect feature-level results.")
            st.download_button("Download sales drift HTML report", data=report_path.read_bytes(),
                               file_name=report_path.name, mime="text/html", key="download_sales_drift_html")
        else:
            st.warning("Sales drift HTML report is missing. Rerun the monitoring script and inspect its console output.")
    with drift_tabs[1]:
        st.write("The current saved diagnostic compares the first 70% and last 30% of rows, not chronological time windows.")
        st.warning("Do not interpret this as confirmed temporal or production drift. A chronological prediction stream is required for that claim.")
        if score_drift:
            d1, d2, d3 = st.columns(3)
            d1.metric("Diagnostic status", str(score_drift.get("status", "UNKNOWN")).replace("_", " ").title())
            d2.metric("Reference rows", f"{int(score_drift.get('reference_rows', 0)):,}")
            d3.metric("Current rows", f"{int(score_drift.get('current_rows', 0)):,}")
            score_dm = score_drift.get("distribution_metrics", {})
            score_metrics = score_drift.get("evidently_result", {}).get("metrics", [])
            score_wasserstein = next((m.get("value") for m in score_metrics
                                      if "ValueDrift" in str(m.get("metric_name", ""))
                                      and "Wasserstein" in str(m.get("metric_name", ""))), None)
            if score_dm:
                st.markdown("**Quantitative score-distribution metrics**")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("KS statistic", display_num(score_dm.get("ks_statistic"), digits=3))
                score_p = score_dm.get("ks_p_value")
                m2.metric("KS p-value", f"{float(score_p):.3g}" if score_p is not None else "N/A")
                m3.metric("PSI", display_num(score_dm.get("population_stability_index"), digits=3))
                m4.metric("Standardized mean shift", display_num(score_dm.get("standardized_mean_shift"), digits=3))
                st.caption(f"Diagnostic heuristic: {str(score_dm.get('status', 'UNKNOWN')).replace('_', ' ').title()}. This is not a temporal drift conclusion.")
            if score_wasserstein is not None:
                st.metric("Evidently normalized Wasserstein distance", display_num(score_wasserstein, digits=3),
                          help="Evidently's saved distribution-distance metric; larger values indicate greater distribution separation under this metric.")
            if not score_dm:
                st.info("KS/PSI/mean-shift metrics are not present in this summary yet. Rerun the updated monitoring script to populate them.")
            if score_drift.get("error"):
                st.error(score_drift["error"])
        report_path = REPORT_DIR / "churnScoreDriftReport.html"
        if report_path.exists():
            st.download_button("Download churn-score diagnostic HTML", data=report_path.read_bytes(),
                               file_name=report_path.name, mime="text/html", key="download_churn_drift_html")
        else:
            st.info("Churn-score drift report is not available yet.")
    with drift_tabs[2]:
        report_path = REPORT_DIR / "modelMonitoringReport.txt"
        if report_path.exists():
            st.text(read_text(report_path))
            st.download_button("Download readable monitoring report", data=report_path.read_bytes(),
                               file_name=report_path.name, mime="text/plain", key="download_monitoring_text")
        else:
            st.info("The readable monitoring report has not been generated yet.")

    limitations = summary.get("limitations", []) if isinstance(summary, dict) else []
    with st.expander("Monitoring limitations and interpretation"):
        st.markdown(
            "- Drift indicates a distribution change; it does not, by itself, prove lower model accuracy.\n"
            "- Forecast metrics reuse the existing holdout and are not a fresh independent evaluation.\n"
            "- Historical source data ends on 9 December 2011; this is not current retail activity.\n"
            "- The churn-score comparison is row-order based and must remain diagnostic-only.\n"
            "- Missing stock-on-hand and unit-cost fields prevent verified inventory impact or savings claims."
        )
        if limitations:
            st.caption("Pipeline notes: " + " · ".join(str(item) for item in limitations))

    st.caption("To refresh these saved diagnostics, run `python src/monitoring/monitorModels.py` from the project root, then reload the app.")
