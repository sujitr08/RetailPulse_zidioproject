from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "processedData" / "cleanedTransactions.csv"
FIGURE_DIR = ROOT / "reports" / "figures"
REPORT_DIR = ROOT / "reports" / "modelReports"
OUTPUT_DIR = ROOT / "data" / "processedData"

RANDOM_STATE = 42
SELECTED_K = 6


def main():
    for directory in [FIGURE_DIR, REPORT_DIR, OUTPUT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing input dataset: {DATA_FILE}")

    print("Loading transactions...")
    df = pd.read_csv(DATA_FILE, low_memory=False)
    df["invoiceDate"] = pd.to_datetime(df["invoiceDate"], errors="coerce")
    df["customerId"] = pd.to_numeric(df["customerId"], errors="coerce")
    df["SalesAmount"] = pd.to_numeric(df["SalesAmount"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

    required = {"invoice", "invoiceDate", "customerId", "SalesAmount", "quantity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Restrict RFM features to plausible sales; keep raw data unchanged.
    if "isCandidateSale" in df.columns:
        sale_mask = df["isCandidateSale"].fillna(False).astype(bool)
    else:
        sale_mask = (
            df["quantity"].gt(0)
            & df["SalesAmount"].gt(0)
            & df["invoiceDate"].notna()
        )

    sales = df.loc[
        sale_mask
        & df["customerId"].notna()
        & df["invoiceDate"].notna()
        & df["SalesAmount"].notna()
    ].copy()

    if sales.empty:
        raise ValueError("No eligible identified-customer sales were found.")

    reference_date = sales["invoiceDate"].max().normalize() + pd.Timedelta(days=1)

    rfm = sales.groupby("customerId").agg(
        lastPurchaseDate=("invoiceDate", "max"),
        frequency=("invoice", "nunique"),
        monetary=("SalesAmount", "sum"),
        totalUnits=("quantity", "sum"),
    )

    rfm["recencyDays"] = (
        reference_date - rfm["lastPurchaseDate"].dt.normalize()
    ).dt.days

    rfm = rfm.reset_index()
    rfm = rfm[
        rfm["recencyDays"].ge(0)
        & rfm["frequency"].gt(0)
        & rfm["monetary"].gt(0)
    ].copy()

    if len(rfm) < 20:
        raise ValueError(
            f"Only {len(rfm)} customers remain; insufficient for reliable clustering."
        )

    # Log transform skewed RFM features, then standardize.
    feature_columns = ["recencyDays", "frequency", "monetary"]
    features = np.log1p(rfm[feature_columns].astype(float))
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    # Compare K-Means solutions rather than assuming a chosen k is best.
    comparison_rows = []
    candidate_k = range(2, min(9, len(rfm) - 1))

    for k in candidate_k:
        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=10,
        )
        labels = model.fit_predict(scaled)
        score = silhouette_score(scaled, labels, random_state=RANDOM_STATE)
        comparison_rows.append({
            "k": k,
            "silhouetteScore": score,
            "inertia": model.inertia_,
        })

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(
        REPORT_DIR / "kMeansEvaluation.csv",
        index=False,
    )

    selected_model = KMeans(
        n_clusters=SELECTED_K,
        random_state=RANDOM_STATE,
        n_init=10,
    )
    rfm["kMeansCluster"] = selected_model.fit_predict(scaled)

    # DBSCAN is a density-based comparison; its parameters are explicit
    # and its result is evaluated separately from K-Means.

    dbscan_results = []
    for eps_value in [0.3, 0.5, 0.7, 0.8, 1.0, 1.2, 1.5]:
        for min_samples_value in [5, 10, 15]:
            candidate_model = DBSCAN(
                eps=eps_value,
                min_samples=min_samples_value,
            )
            candidate_labels = candidate_model.fit_predict(scaled)

            non_noise_mask = candidate_labels != -1
            non_noise_labels = candidate_labels[non_noise_mask]
            cluster_count = len(set(non_noise_labels))
            noise_count = int((candidate_labels == -1).sum())
            noise_percentage = noise_count / len(candidate_labels) * 100

            cluster_sizes = pd.Series(non_noise_labels).value_counts()
            smallest_cluster_size = (
                int(cluster_sizes.min()) if cluster_count else 0
            )

            candidate_silhouette = np.nan
            if (
                cluster_count >= 2
                and len(non_noise_labels) > cluster_count
            ):
                candidate_silhouette = silhouette_score(
                    scaled[non_noise_mask],
                    non_noise_labels,
                )

            dbscan_results.append({
                "eps": eps_value,
                "minSamples": min_samples_value,
                "clusterCount": cluster_count,
                "noiseCount": noise_count,
                "noisePercentage": round(noise_percentage, 2),
                "smallestClusterSize": smallest_cluster_size,
                "silhouetteScore": candidate_silhouette,
            })

    dbscan_evaluation = pd.DataFrame(dbscan_results)
    dbscan_evaluation.to_csv(
        REPORT_DIR / "dbscanParameterEvaluation.csv",
        index=False,
    )

    # Exclude configurations with excessive noise or tiny clusters.
    valid_candidates = dbscan_evaluation[
        (dbscan_evaluation["clusterCount"] >= 2)
        & (dbscan_evaluation["noisePercentage"] <= 50)
        & (dbscan_evaluation["smallestClusterSize"] >= 20)
        & (dbscan_evaluation["silhouetteScore"].notna())
    ]

    if not valid_candidates.empty:
        best_candidate = valid_candidates.sort_values(
            "silhouetteScore",
            ascending=False,
        ).iloc[0]

        selected_eps = float(best_candidate["eps"])
        selected_min_samples = int(best_candidate["minSamples"])
    else:
        # Keep the original parameters if no tested configuration qualifies.
        selected_eps = 0.8
        selected_min_samples = 10

    dbscan_model = DBSCAN(
        eps=selected_eps,
        min_samples=selected_min_samples,
    )
    rfm["dbscanCluster"] = dbscan_model.fit_predict(scaled)

    non_noise = rfm["dbscanCluster"] != -1
    dbscan_clusters = rfm.loc[non_noise, "dbscanCluster"].nunique()
    dbscan_noise_count = int((rfm["dbscanCluster"] == -1).sum())
    dbscan_noise_percentage = dbscan_noise_count / len(rfm) * 100
    dbscan_score = np.nan

    if dbscan_clusters >= 2 and non_noise.sum() > dbscan_clusters:
        dbscan_score = silhouette_score(
            scaled[non_noise.to_numpy()],
            rfm.loc[non_noise, "dbscanCluster"],
        )

    print("\nDBSCAN PARAMETER COMPARISON")
    print(
        dbscan_evaluation.sort_values(
            "silhouetteScore",
            ascending=False,
            na_position="last",
        ).to_string(index=False)
    )
    print(f"\nSelected DBSCAN eps: {selected_eps}")
    print(f"Selected DBSCAN min_samples: {selected_min_samples}")
    print(f"DBSCAN noise percentage: {dbscan_noise_percentage:.2f}%")


    # PCA for a 2D visualization of the standardized RFM feature space.
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    components = pca.fit_transform(scaled)
    rfm["pca1"] = components[:, 0]
    rfm["pca2"] = components[:, 1]

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        rfm["pca1"],
        rfm["pca2"],
        c=rfm["kMeansCluster"],
        cmap="tab10",
        s=18,
        alpha=0.75,
    )
    ax.set_title("Customer Segmentation — K-Means in PCA Space")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend(
        *scatter.legend_elements(),
        title="K-Means cluster",
        loc="best",
    )
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "customerSegmentationPCA.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        rfm["pca1"],
        rfm["pca2"],
        c=rfm["dbscanCluster"],
        cmap="tab10",
        s=18,
        alpha=0.75,
    )
    ax.set_title("Customer Segmentation — DBSCAN in PCA Space")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    fig.colorbar(scatter, ax=ax, label="DBSCAN cluster (-1 = noise)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "customerSegmentationDBSCANPCA.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(comparison["k"], comparison["silhouetteScore"], marker="o")
    ax.set_title("K-Means Cluster Count Evaluation")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_xticks(comparison["k"])
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "kMeansSilhouetteScores.png", dpi=160)
    plt.close(fig)

    variance = pca.explained_variance_ratio_

    cluster_summary = rfm.groupby("kMeansCluster").agg(
        customerCount=("customerId", "size"),
        medianRecencyDays=("recencyDays", "median"),
        medianFrequency=("frequency", "median"),
        medianMonetary=("monetary", "median"),
        totalMonetary=("monetary", "sum"),
    ).reset_index()

    # Save assigned customers and a readable segment summary.
    rfm.to_csv(OUTPUT_DIR / "customerSegments.csv", index=False)
    cluster_summary.to_csv(
        REPORT_DIR / "customerSegmentSummary.csv",
        index=False,
    )
    
    dbscan_summary = rfm.groupby("dbscanCluster").agg(
        customerCount=("customerId", "size"),
        medianRecencyDays=("recencyDays", "median"),
        medianFrequency=("frequency", "median"),
        medianMonetary=("monetary", "median"),
        totalMonetary=("monetary", "sum"),
    ).reset_index()

    dbscan_summary.to_csv(
        REPORT_DIR / "dbscanClusterSummary.csv",
        index=False,
    )

    report = [
        "RETAILPULSE — CUSTOMER SEGMENTATION REPORT",
        "=" * 55,
        f"Eligible identified customers: {len(rfm):,}",
        f"RFM reference date: {reference_date.date()}",
        "RFM features: recencyDays, frequency, monetary",
        "Transform: log1p followed by StandardScaler",
        f"Highest silhouette score: k={int(comparison.loc[comparison['silhouetteScore'].idxmax(), 'k'])}",
        f"Selection rationale: six segments retained for business interpretation; "
        f"the highest silhouette score alone favors fewer clusters.",
        f"Selected K-Means silhouette: "
        f"{silhouette_score(scaled, rfm['kMeansCluster']):.4f}",
        f"Selected DBSCAN eps: {selected_eps}",
        f"Selected DBSCAN min_samples: {selected_min_samples}",
        f"DBSCAN clusters excluding noise: {dbscan_clusters}",
        f"DBSCAN noise customers: {dbscan_noise_count:,}",
        f"DBSCAN noise percentage: {dbscan_noise_percentage:.2f}%",
        f"DBSCAN silhouette excluding noise: {dbscan_score}",
        f"PCA component 1 variance: {variance[0]:.4%}",
        f"PCA component 2 variance: {variance[1]:.4%}",
        f"PCA combined explained variance: {variance.sum():.4%}",
        "",
        "K-MEANS EVALUATION ACROSS K",
        comparison.to_string(index=False),
        "",
        "CUSTOMER SEGMENT SUMMARY",
        cluster_summary.to_string(index=False),
        "",
        "LIMITATIONS",
        "- Cluster IDs are numerical labels, not inherent customer types.",
        "- Business names for segments require interpretation of their RFM profiles.",
        "- DBSCAN results depend on eps and min_samples.",
        "- PCA is a projection; it can hide separation present in higher dimensions.",
        "- Monetary value reflects the source currency and the chosen sale filter.",
        "- RFM is based on this dataset's observed history, not future behavior.",
    ]

    (REPORT_DIR / "customerSegmentationReport.txt").write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("\nSegmentation completed.")
    print(f"Eligible customers: {len(rfm):,}")
    print(f"K-Means clusters: {SELECTED_K}")
    print(f"DBSCAN parameters: eps={selected_eps}, min_samples={selected_min_samples}")
    print(f"DBSCAN clusters (excluding noise): {dbscan_clusters}")
    print(f"DBSCAN noise percentage: {dbscan_noise_percentage:.2f}%")
    print(f"Combined PCA explained variance: {variance.sum():.2%}")
    print(f"Report: {REPORT_DIR / 'customerSegmentationReport.txt'}")
    print(f"Customer output: {OUTPUT_DIR / 'customerSegments.csv'}")


if __name__ == "__main__":
    main()
