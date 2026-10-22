
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# ========================================
# 1. PROJECT CONFIGURATION
# ========================================

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
RESULTS = BASE / "results"

RESULTS.mkdir(exist_ok=True)

START_MONTH = "2017-01"
END_MONTH = "2018-07"

MONTHS = pd.period_range(
    START_MONTH, END_MONTH, freq="M"
)

K = 4
RANDOM_SEED = 42


def run():

    # ========================================
    # 2. LOAD THE DATASET
    # ========================================

    required_files = [
        "olist_orders_dataset.csv",
        "olist_order_items_dataset.csv",
        "olist_products_dataset.csv",
        "product_category_name_translation.csv",
    ]

    missing = [
        name for name in required_files
        if not (DATA / name).exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing files in data/: "
            + ", ".join(missing)
        )

    orders = pd.read_csv(
        DATA / required_files[0],
        usecols=[
            "order_id",
            "order_status",
            "order_purchase_timestamp"
        ]
    )

    items = pd.read_csv(
        DATA / required_files[1],
        usecols=[
            "order_id",
            "product_id",
            "price"
        ]
    )

    products = pd.read_csv(
        DATA / required_files[2],
        usecols=[
            "product_id",
            "product_category_name"
        ]
    )

    translation = pd.read_csv(
        DATA / required_files[3]
    )

    print("Orders loaded:", len(orders))
    print("Order items loaded:", len(items))

    # ========================================
    # 3. DATA PREPROCESSING
    # ========================================

    orders["order_purchase_timestamp"] = (
        pd.to_datetime(
            orders["order_purchase_timestamp"],
            errors="coerce"
        )
    )

    orders = orders.drop_duplicates("order_id")

    # Merge items with their corresponding orders
    sales = items.merge(
        orders,
        on="order_id",
        how="inner",
        validate="many_to_one"
    )

    # Keep delivered orders within the chosen period
    sales = sales.loc[
        (sales["order_status"] == "delivered")
        & (
            sales["order_purchase_timestamp"]
            >= pd.Timestamp("2017-01-01")
        )
        & (
            sales["order_purchase_timestamp"]
            < pd.Timestamp("2018-08-01")
        )
        & sales["product_id"].notna()
    ].copy()

    sales["month"] = (
        sales["order_purchase_timestamp"]
        .dt.to_period("M")
    )

    print("Filtered order items:", len(sales))
    print(
        "Unique products:",
        sales["product_id"].nunique()
    )

    if sales["product_id"].nunique() < 7:
        raise ValueError(
            "Not enough products for clustering."
        )

    # ========================================
    # 4. MONTHLY PRODUCT DEMAND
    # ========================================

    # Every order-item row represents one sold item
    monthly = (
        sales.groupby(["product_id", "month"])
        .size()
        .unstack(fill_value=0)
    )

    # Include months with no sales
    monthly = monthly.reindex(
        columns=MONTHS,
        fill_value=0
    )

    monthly.index.name = "product_id"

    monthly.to_csv(
        RESULTS / "monthly_product_sales.csv"
    )

    # ========================================
    # 5. FEATURE ENGINEERING
    # ========================================

    a = monthly.to_numpy(dtype=float)

    totals = a.sum(axis=1)
    mean = a.mean(axis=1)
    std = a.std(axis=1)

    active_months = (a > 0).sum(axis=1)

    last_positions = np.where(
        a > 0,
        np.arange(a.shape[1]),
        -1
    ).max(axis=1)

    recency_months = (
        a.shape[1] - 1 - last_positions
    )

    features = pd.DataFrame({
        "product_id": monthly.index,
        "total_units": totals.astype(int),
        "avg_monthly_units": mean,
        "std_monthly_units": std,
        "active_months": active_months,
        "recency_months": recency_months,
        "coefficient_of_variation":
            std / (mean + 0.2),
        "peak_month_share":
            a.max(axis=1) / totals
    })

    features = features.set_index("product_id")

    print("\nDemand features:")
    print(features.head())

    # ========================================
    # 6. SELECT FEATURES FOR K-MEANS
    # ========================================

    # Log transform reduces extreme differences
    # between high-volume and low-volume products.
    X = pd.DataFrame(index=features.index)

    X["log_total_units"] = np.log1p(
        features["total_units"]
    )

    X["active_months"] = features["active_months"]

    X["recency_months"] = (
        features["recency_months"]
    )

    X["coefficient_of_variation"] = (
        features["coefficient_of_variation"]
    )

    X["peak_month_share"] = (
        features["peak_month_share"]
    )

    # Scale values to comparable ranges
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)

    # ========================================
    # 7. FIND A SUITABLE K VALUE
    # ========================================

    scores = []

    for k in range(2, 7):

        candidate = KMeans(
            n_clusters=k,
            random_state=RANDOM_SEED,
            n_init=10
        )

        assigned = candidate.fit_predict(scaled)

        silhouette = silhouette_score(
            scaled,
            assigned,
            sample_size=min(1500, len(scaled)),
            random_state=RANDOM_SEED
        )

        scores.append([
            k,
            candidate.inertia_,
            silhouette
        ])

    score_df = pd.DataFrame(
        scores,
        columns=[
            "k",
            "inertia",
            "silhouette_score"
        ]
    )

    print("\nClustering evaluation:")
    print(score_df)

    score_df.to_csv(
        RESULTS / "clustering_evaluation.csv",
        index=False
    )

    # ========================================
    # 8. APPLY K-MEANS CLUSTERING
    # ========================================

    model = KMeans(
        n_clusters=K,
        random_state=RANDOM_SEED,
        n_init=10
    )

    features["cluster"] = model.fit_predict(scaled)

    print("\nProduct count per cluster:")
    print(features["cluster"].value_counts())

    # ========================================
    # 9. ANALYZE CLUSTER CHARACTERISTICS
    # ========================================

    profile = features.groupby("cluster").agg(
        products=("total_units", "size"),
        median_total_units=(
            "total_units", "median"
        ),
        median_monthly_units=(
            "avg_monthly_units", "median"
        ),
        median_active_months=(
            "active_months", "median"
        ),
        median_cv=(
            "coefficient_of_variation", "median"
        ),
        median_recency_months=(
            "recency_months", "median"
        ),
        median_peak_share=(
            "peak_month_share", "median"
        )
    )

    # K-Means clusters have numeric IDs.
    # The names below are business interpretations.

    rank = profile.sort_values(
        [
            "median_total_units",
            "median_active_months"
        ],
        ascending=[False, False]
    ).index.tolist()

    middle = rank[1:3]

    volatile = profile.loc[
        middle, "median_cv"
    ].idxmax()

    moderate = next(
        c for c in middle if c != volatile
    )

    labels = {
        rank[0]: "Higher demand",
        rank[-1]: "Lower demand",
        volatile: "Irregular demand",
        moderate: "Moderate demand"
    }

    strategies = {
        "Higher demand":
            "Review stock weekly and replenish frequently",

        "Moderate demand":
            "Use periodic replenishment and monitor sales",

        "Irregular demand":
            "Review demand spikes; keep flexible inventory",

        "Lower demand":
            "Use small purchases; avoid excess stock"
    }

    profile["business_label"] = (
        profile.index.map(labels)
    )

    profile["inventory_strategy"] = (
        profile["business_label"].map(strategies)
    )

    profile.to_csv(
        RESULTS / "cluster_summary.csv"
    )

    features["business_label"] = (
        features["cluster"].map(labels)
    )

    features["inventory_strategy"] = (
        features["business_label"].map(strategies)
    )

    print("\nCluster summary:")
    print(profile)

    # ========================================
    # 10. ADD CATEGORY AND REVENUE DETAILS
    # ========================================

    products = products.drop_duplicates(
        "product_id"
    )

    translation = translation.drop_duplicates(
        "product_category_name"
    )

    products = products.merge(
        translation,
        on="product_category_name",
        how="left"
    )

    products["category"] = (
        products["product_category_name_english"]
        .fillna(products["product_category_name"])
        .fillna("Unknown")
    )

    revenue = (
        sales.groupby("product_id")["price"]
        .sum()
        .rename("historical_revenue_brl")
    )

    product_result = (
        features.join(revenue, how="left")
        .join(
            products.set_index("product_id")[["category"]],
            how="left"
        )
    )

    product_result["category"] = (
        product_result["category"].fillna("Unknown")
    )

    # ========================================
    # 11. ILLUSTRATIVE REORDER POINT
    # ========================================

    # Assumed supplier lead time: 14 days
    # Assumed service factor: 1.65
    #
    # This is only an approximate demonstration.
    # The dataset does NOT provide actual inventory
    # levels or supplier lead times.

    lead_days = 14
    z = 1.65

    reorder_point = (
        product_result["avg_monthly_units"]
        * lead_days / 30.0
        +
        z
        * product_result["std_monthly_units"]
        * np.sqrt(lead_days / 30.0)
    )

    product_result[
        "illustrative_reorder_point_14d"
    ] = np.ceil(reorder_point).astype(int)

    product_result.reset_index().to_csv(
        RESULTS / "product_clusters.csv",
        index=False
    )

    # ========================================
    # 12. ELBOW AND SILHOUETTE GRAPHS
    # ========================================

    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4)
    )

    axes[0].plot(
        score_df["k"],
        score_df["inertia"],
        marker="o"
    )

    axes[0].set_title("Elbow Method")
    axes[0].set_xlabel("Number of Clusters (K)")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(
        score_df["k"],
        score_df["silhouette_score"],
        marker="o"
    )

    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("Number of Clusters (K)")
    axes[1].set_ylabel("Silhouette Score")

    plt.tight_layout()

    plt.savefig(
        RESULTS / "cluster_evaluation.png",
        dpi=170
    )

    plt.close()

    # ========================================
    # 13. VISUALIZE CLUSTERS USING PCA
    # ========================================

    pca = PCA(
        n_components=2,
        random_state=RANDOM_SEED
    )

    points = pca.fit_transform(scaled)

    plot_df = pd.DataFrame({
        "PC1": points[:, 0],
        "PC2": points[:, 1],
        "Segment":
            features["business_label"].to_numpy()
    })

    # Limit plotting size for readability
    if len(plot_df) > 5000:
        plot_df = plot_df.sample(
            5000,
            random_state=RANDOM_SEED
        )

    plt.figure(figsize=(9, 5))

    sns.scatterplot(
        data=plot_df,
        x="PC1",
        y="PC2",
        hue="Segment",
        s=16,
        alpha=0.6
    )

    plt.title(
        "Product Demand Clusters - PCA Visualization"
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS / "product_cluster_plot.png",
        dpi=170
    )

    plt.close()

    # ========================================
    # 14. PLOT CLUSTER DISTRIBUTION
    # ========================================

    plt.figure(figsize=(8, 4))

    profile.sort_values(
        "products",
        ascending=False
    )["products"].plot(kind="bar")

    plt.title("Number of Products per Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Products")

    plt.tight_layout()

    plt.savefig(
        RESULTS / "products_by_cluster.png",
        dpi=170
    )

    plt.close()

    # ========================================
    # 15. COMPLETE
    # ========================================

    print("\nClustering completed successfully!")
    print("All outputs saved in:", RESULTS)


if __name__ == "__main__":
    run()
