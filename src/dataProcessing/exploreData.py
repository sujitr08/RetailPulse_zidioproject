from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "processedData" / "cleanedTransactions.csv"
FIGURE_DIR = ROOT / "reports" / "figures"
REPORT_DIR = ROOT / "reports" / "modelReports"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    print("Loading cleaned transactions...")
    df = pd.read_csv(DATA_FILE, low_memory=False)

    required_columns = {
        "invoice", "stockCode", "quantity", "invoiceDate",
        "unitPrice", "customerId", "country", "SalesAmount"
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df["invoiceDate"] = pd.to_datetime(df["invoiceDate"], errors="coerce")
    for column in ["quantity", "unitPrice", "SalesAmount"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    report = []
    report.append("RETAILPULSE — EXPLORATORY DATA ANALYSIS")
    report.append("=" * 55)
    report.append(f"Dataset rows: {len(df):,}")
    report.append(f"Dataset columns: {len(df.columns)}")
    report.append(f"Date range: {df['invoiceDate'].min()} to {df['invoiceDate'].max()}")
    report.append(f"Unique invoices: {df['invoice'].nunique():,}")
    report.append(f"Unique stock codes: {df['stockCode'].nunique():,}")
    report.append(f"Unique identified customers: {df['customerId'].nunique():,}")
    report.append(f"Countries: {df['country'].nunique():,}")
    report.append("")

    report.append("MISSING VALUES")
    report.append("-" * 55)
    missing = df.isna().sum().sort_values(ascending=False)
    for column, count in missing.items():
        report.append(
            f"{column}: {count:,} ({count / len(df) * 100:.2f}%)"
        )

    report.append("")
    report.append("NUMERIC SUMMARY")
    report.append("-" * 55)
    report.append(
        df[["quantity", "unitPrice", "SalesAmount"]]
        .describe()
        .round(2)
        .to_string()
    )

    # This is an analysis subset, not a replacement for the full dataset.
    if "isCandidateSale" in df.columns:
        candidate_mask = df["isCandidateSale"].fillna(False).astype(bool)
    else:
        candidate_mask = (
            df["quantity"].gt(0)
            & df["unitPrice"].gt(0)
            & df["invoiceDate"].notna()
        )

    sales = df.loc[candidate_mask].copy()
    sales["salesMonth"] = sales["invoiceDate"].dt.to_period("M").astype(str)

    report.append("")
    report.append("PRELIMINARY CANDIDATE-SALE SUBSET")
    report.append("-" * 55)
    report.append(f"Rows selected: {len(sales):,}")
    report.append(
        "Note: candidate-sale filtering is preliminary and requires "
        "business-rule validation."
    )
    report.append(
        f"Candidate subset sales amount: {sales['SalesAmount'].sum():,.2f}"
    )

    # 1. Missing-value percentages
    missing_pct = (df.isna().mean() * 100).sort_values()
    fig, ax = plt.subplots(figsize=(10, 7))
    missing_pct[missing_pct > 0].plot(kind="barh", ax=ax)
    ax.set_title("Missing Values by Column")
    ax.set_xlabel("Missing values (%)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "missingValues.png", dpi=160)
    plt.close(fig)

    # 2. Quantity distribution, restricted to central 99% for readability.
    quantity = df["quantity"].dropna()
    if not quantity.empty:
        lower, upper = quantity.quantile([0.01, 0.99])
        visible = quantity[quantity.between(lower, upper)]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(visible, bins=60)
        ax.set_title("Quantity Distribution (Central 98% Range)")
        ax.set_xlabel("Quantity")
        ax.set_ylabel("Transaction-line count")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "quantityDistribution.png", dpi=160)
        plt.close(fig)

    # 3. Unit-price distribution, restricted to central 99% for readability.
    prices = df["unitPrice"].dropna()
    if not prices.empty:
        lower, upper = prices.quantile([0.01, 0.99])
        visible = prices[prices.between(lower, upper)]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(visible, bins=60)
        ax.set_title("Unit Price Distribution (Central 98% Range)")
        ax.set_xlabel("Unit price")
        ax.set_ylabel("Transaction-line count")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "unitPriceDistribution.png", dpi=160)
        plt.close(fig)

    # 4. Monthly candidate sales
    monthly = sales.groupby("salesMonth")["SalesAmount"].sum().sort_index()
    if not monthly.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(monthly.index, monthly.values, marker="o", markersize=3)
        ax.set_title("Monthly Candidate-Sale Amount")
        ax.set_xlabel("Month")
        ax.set_ylabel("Sales amount (source currency)")
        ax.tick_params(axis="x", rotation=60)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "monthlySales.png", dpi=160)
        plt.close(fig)
        report.append("")
        report.append("MONTHLY CANDIDATE-SALE AMOUNT")
        report.append("-" * 55)
        report.append(monthly.round(2).to_string())

    # 5. Top countries by candidate-sale amount
    if not sales.empty:
        country_sales = (
            sales.groupby("country")["SalesAmount"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .sort_values()
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        country_sales.plot(kind="barh", ax=ax)
        ax.set_title("Top 10 Countries by Candidate-Sale Amount")
        ax.set_xlabel("Sales amount (source currency)")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "topCountriesBySales.png", dpi=160)
        plt.close(fig)

    # 6. Correlation heatmap for numeric transaction measures
    numeric_columns = ["quantity", "unitPrice", "SalesAmount"]
    corr = df[numeric_columns].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(corr, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(numeric_columns)), labels=numeric_columns, rotation=30)
    ax.set_yticks(range(len(numeric_columns)), labels=numeric_columns)
    for i in range(len(numeric_columns)):
        for j in range(len(numeric_columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
    ax.set_title("Correlation Heatmap — Numeric Measures")
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "correlationHeatmap.png", dpi=160)
    plt.close(fig)

    # 7. Product categories, including the uncategorized group
    if "ProductCategory" in df.columns:
        categories = df["ProductCategory"].fillna("Missing").value_counts().head(12)
        fig, ax = plt.subplots(figsize=(10, 6))
        categories.sort_values().plot(kind="barh", ax=ax)
        ax.set_title("Transaction Rows by Product Category")
        ax.set_xlabel("Transaction-line count")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "productCategories.png", dpi=160)
        plt.close(fig)

        report.append("")
        report.append("PRODUCT CATEGORY COVERAGE")
        report.append("-" * 55)
        report.append(df["ProductCategory"].fillna("Missing").value_counts().to_string())

    report.append("")
    report.append("INTERPRETATION AND LIMITATIONS")
    report.append("-" * 55)
    report.append(
        "- Candidate-sale results depend on the current filtering rule."
    )
    report.append(
        "- Product categories are heuristic labels, not verified product taxonomy."
    )
    report.append(
        "- Missing customer IDs limit customer-level analysis."
    )
    report.append(
        "- The numeric correlation heatmap describes association, not causation."
    )
    report.append(
        "- Source data does not provide actual unit costs or stock-on-hand quantities."
    )

    report_path = REPORT_DIR / "edaReport.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print("\nEDA completed.")
    print(f"Rows analyzed: {len(df):,}")
    print(f"Candidate-sale rows: {len(sales):,}")
    print(f"Report: {report_path}")
    print(f"Figures: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
