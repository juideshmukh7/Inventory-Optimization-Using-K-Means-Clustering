
from pathlib import Path

import pandas as pd
import streamlit as st


# ========================================
# 1. LOAD MODEL OUTPUT
# ========================================

BASE = Path(__file__).resolve().parent

FILE = BASE / "results" / "product_clusters.csv"

st.set_page_config(
    page_title="Inventory Optimization",
    layout="wide"
)

st.title(
    "Inventory Optimization Using Product Demand Clustering"
)

st.caption(
    "DWML Mini Project | K-Means Clustering | "
    "Olist E-Commerce Dataset"
)

if not FILE.exists():
    st.error(
        "Run analysis.py first to generate clustering results."
    )
    st.stop()


@st.cache_data
def load_data():
    return pd.read_csv(FILE)


df = load_data()


# ========================================
# 2. SIDEBAR FILTERS
# ========================================

st.sidebar.header("Filters")

categories = (
    ["All Categories"]
    + sorted(df["category"].fillna("Unknown").unique())
)

selected_category = st.sidebar.selectbox(
    "Select Product Category",
    categories
)

segments = (
    ["All Segments"]
    + sorted(df["business_label"].unique())
)

selected_segment = st.sidebar.selectbox(
    "Select Demand Segment",
    segments
)

filtered = df.copy()

if selected_category != "All Categories":
    filtered = filtered[
        filtered["category"] == selected_category
    ]

if selected_segment != "All Segments":
    filtered = filtered[
        filtered["business_label"] == selected_segment
    ]


# ========================================
# 3. KEY PERFORMANCE INDICATORS
# ========================================

st.subheader("Business Overview")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Products Analyzed",
    f"{len(filtered):,}"
)

col2.metric(
    "Units Sold",
    f"{int(filtered['total_units'].sum()):,}"
)

col3.metric(
    "Historical Revenue (BRL)",
    f"{filtered['historical_revenue_brl'].sum():,.2f}"
)


# ========================================
# 4. CLUSTER DISTRIBUTION
# ========================================

st.subheader("Demand Clustering Results")

left, right = st.columns(2)

with left:
    st.write("Number of Products per Segment")

    counts = (
        filtered.groupby("business_label")
        .size()
    )

    st.bar_chart(counts)

with right:
    st.write("Median Sales per Product")

    sales = (
        filtered.groupby("business_label")[
            "total_units"
        ].median()
    )

    st.bar_chart(sales)


# ========================================
# 5. PRODUCT EXPLORER
# ========================================

st.subheader("Product Demand Analysis")

columns = [
    "product_id",
    "category",
    "total_units",
    "active_months",
    "recency_months",
    "coefficient_of_variation",
    "business_label",
    "inventory_strategy"
]

st.dataframe(
    filtered[columns].sort_values(
        "total_units",
        ascending=False
    ),
    use_container_width=True,
    hide_index=True
)


# ========================================
# 6. INVENTORY REORDER DEMONSTRATION
# ========================================

st.subheader("Inventory Reorder Demonstration")

st.info(
    "This is an illustrative calculation. "
    "Actual stock levels and supplier lead times "
    "are not present in the Olist dataset."
)

if len(filtered) > 0:

    product_id = st.selectbox(
        "Select Product ID",
        filtered["product_id"].astype(str).tolist()
    )

    product = filtered[
        filtered["product_id"].astype(str)
        == product_id
    ].iloc[0]

    st.write(
        "**Category:**",
        product["category"]
    )

    st.write(
        "**Demand Segment:**",
        product["business_label"]
    )

    st.write(
        "**Suggested Inventory Strategy:**",
        product["inventory_strategy"]
    )

    st.write(
        "**Total Historical Units Sold:**",
        int(product["total_units"])
    )

    reorder_point = int(
        product["illustrative_reorder_point_14d"]
    )

    st.write(
        "**Illustrative Reorder Point:**",
        reorder_point,
        "units"
    )

    current_stock = st.number_input(
        "Enter Assumed Current Stock",
        min_value=0,
        value=0,
        step=1
    )

    if current_stock <= reorder_point:

        st.warning(
            "Stock is at or below the illustrative "
            "reorder point. Review replenishment."
        )

        st.write(
            "Indicative shortfall:",
            max(0, reorder_point - current_stock),
            "units"
        )

    else:

        st.success(
            "Current stock is above the illustrative "
            "reorder point."
        )

else:
    st.info("No products match the selected filters.")


# ========================================
# 7. FINAL MESSAGE
# ========================================

st.divider()

st.caption(
    "Academic demonstration using historical "
    "fulfilled sales, not real-time warehouse inventory."
)
