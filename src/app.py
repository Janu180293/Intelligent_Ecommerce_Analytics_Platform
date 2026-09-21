"""
Phase 10 - Streamlit Dashboard
Intelligent E-Commerce Customer Analytics Platform (Olist dataset)

Run:
    streamlit run app.py
Expects the ./outputs/ folder (produced by scripts 01-09) to sit alongside this file,
OR pass a different path via the OUTPUTS_DIR environment variable.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib
from PIL import Image

st.set_page_config(page_title="E-Commerce Customer Analytics Platform", layout="wide", page_icon="🛍️")

OUT = os.environ.get("OUTPUTS_DIR", "outputs/")


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_orders():
    return pd.read_csv(OUT + "orders_master_clean.csv", parse_dates=["order_purchase_timestamp"])


@st.cache_data
def load_items():
    return pd.read_csv(OUT + "items_enriched.csv")


@st.cache_data
def load_segments():
    return pd.read_csv(OUT + "customer_segments.csv")


@st.cache_data
def load_purchase_pred_dataset():
    return pd.read_csv(OUT + "purchase_prediction_dataset.csv")


@st.cache_data
def load_clv_dataset():
    return pd.read_csv(OUT + "clv_prediction_dataset.csv")


@st.cache_data
def load_product_stats():
    return pd.read_csv(OUT + "product_stats.csv")


@st.cache_data
def load_recommendations():
    return pd.read_csv(OUT + "customer_recommendations_precomputed.csv")


@st.cache_resource
def load_purchase_model():
    pipeline = joblib.load(OUT + "purchase_prediction_pipeline.joblib")
    cols = joblib.load(OUT + "purchase_prediction_feature_cols.joblib")
    return pipeline, cols


@st.cache_resource
def load_clv_model():
    pipeline = joblib.load(OUT + "clv_prediction_pipeline.joblib")
    cols = joblib.load(OUT + "clv_prediction_feature_cols.joblib")
    return pipeline, cols


@st.cache_data
def load_model_comparisons():
    clf = pd.read_csv(OUT + "clf_model_comparison.csv")
    clv = pd.read_csv(OUT + "clv_model_comparison.csv")
    reco = pd.read_csv(OUT + "reco_evaluation.csv")
    return clf, clv, reco


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🛍️ Analytics Platform")
page = st.sidebar.radio("Navigate", [
    "Executive Dashboard",
    "Customer Analytics",
    "Recommendation Engine",
    "Business Insights",
    "Explainability",
])
st.sidebar.markdown("---")
st.sidebar.caption("Intelligent E-Commerce Customer Analytics Platform · Olist dataset")

# ---------------------------------------------------------------------------
# PAGE 1 — Executive Dashboard
# ---------------------------------------------------------------------------
if page == "Executive Dashboard":
    st.title("Executive Dashboard")

    orders = load_orders()
    delivered = orders[orders["order_status"] == "delivered"]
    segments = load_segments()

    total_revenue = delivered["total_payment_value"].sum()
    total_customers = segments["customer_unique_id"].nunique()
    total_orders = delivered["order_id"].nunique()
    aov = delivered["total_payment_value"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"R$ {total_revenue:,.0f}")
    c2.metric("Total Customers", f"{total_customers:,}")
    c3.metric("Total Orders", f"{total_orders:,}")
    c4.metric("Avg Order Value", f"R$ {aov:,.2f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Monthly Revenue Trend")
        monthly = (delivered.set_index("order_purchase_timestamp")
                   .resample("MS")["total_payment_value"].sum().reset_index())
        fig = px.line(monthly, x="order_purchase_timestamp", y="total_payment_value", markers=True)
        fig.update_layout(xaxis_title="Month", yaxis_title="Revenue (R$)")
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.subheader("Customer Segment Distribution")
        seg_counts = segments["segment"].value_counts().reset_index()
        seg_counts.columns = ["segment", "count"]
        fig = px.pie(seg_counts, names="segment", values="count", hole=0.4)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Orders by State (Top 15)")
    state_counts = orders["customer_state"].value_counts().head(15).reset_index()
    state_counts.columns = ["state", "orders"]
    fig = px.bar(state_counts, x="state", y="orders")
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# PAGE 2 — Customer Analytics
# ---------------------------------------------------------------------------
elif page == "Customer Analytics":
    st.title("Customer Analytics")

    segments = load_segments()
    purchase_data = load_purchase_pred_dataset()
    clv_data = load_clv_dataset()

    customer_id = st.selectbox(
        "Select a customer (customer_unique_id)",
        options=segments["customer_unique_id"].head(500).tolist(),
        help="Showing first 500 for demo speed — in production this would be a search box.",
    )
    custom_input = st.text_input("...or paste any customer_unique_id directly")
    if custom_input:
        customer_id = custom_input.strip()

    seg_row = segments[segments["customer_unique_id"] == customer_id]
    if seg_row.empty:
        st.error("Customer not found in the segmentation dataset.")
    else:
        seg_row = seg_row.iloc[0]
        st.markdown(f"### Segment: **{seg_row['segment']}**")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Recency (days)", f"{seg_row['recency_days']:.0f}")
        c2.metric("Frequency (orders)", f"{seg_row['frequency']:.0f}")
        c3.metric("Monetary (R$)", f"{seg_row['monetary']:.2f}")
        c4.metric("Avg Order Value", f"R$ {seg_row['avg_order_value']:.2f}")

        st.markdown("---")
        col1, col2 = st.columns(2)

        # --- Purchase probability ---
        with col1:
            st.subheader("Purchase Probability (next 90 days)")
            row = purchase_data[purchase_data["customer_unique_id"] == customer_id]
            if row.empty:
                st.info("This customer has no pre-cutoff order history eligible for the "
                         "point-in-time purchase-prediction model (see Phase 6 methodology).")
            else:
                pipeline, feat_cols = load_purchase_model()
                X = row[feat_cols + ["customer_state"]]
                proba = pipeline.predict_proba(X)[0, 1]
                st.metric("Predicted probability", f"{proba:.1%}")
                st.progress(min(float(proba), 1.0))
                st.caption("⚠️ Model AUC is ~0.5-0.57 (see Phase 6 notebook) — repeat purchase "
                           "within a fixed 90-day window is a genuinely rare, hard-to-predict event "
                           "in this dataset. Treat this as a weak ranking signal, not a precise probability.")

        # --- CLV prediction ---
        with col2:
            st.subheader("Predicted Customer Lifetime Value (next 90 days)")
            row = clv_data[clv_data["customer_unique_id"] == customer_id]
            if row.empty:
                st.info("This customer has no pre-cutoff order history eligible for the "
                         "point-in-time CLV model (see Phase 7 methodology).")
            else:
                pipeline, feat_cols = load_clv_model()
                X = row[feat_cols + ["customer_state"]]
                pred_log = pipeline.predict(X)[0]
                pred = max(np.expm1(pred_log), 0)
                st.metric("Predicted future spend", f"R$ {pred:.2f}")
                st.caption("⚠️ R² ≈ 0 on this target (see Phase 7 notebook) — ~99% of customers spend "
                           "R$0 in any given 90-day window in this dataset, so treat this as a rough "
                           "ranking signal rather than a precise forecast.")

        st.markdown("---")
        st.subheader("Full Customer Profile")
        st.dataframe(seg_row.to_frame().T, width="stretch")

# ---------------------------------------------------------------------------
# PAGE 3 — Recommendation Engine
# ---------------------------------------------------------------------------
elif page == "Recommendation Engine":
    st.title("Recommendation Engine")

    recos = load_recommendations()
    product_stats = load_product_stats()

    customer_id = st.selectbox(
        "Select a customer (customer_unique_id)",
        options=recos["customer_unique_id"].head(500).tolist(),
    )
    custom_input = st.text_input("...or paste any customer_unique_id directly", key="reco_input")
    if custom_input:
        customer_id = custom_input.strip()

    row = recos[recos["customer_unique_id"] == customer_id]
    if row.empty:
        st.info("No precomputed recommendations for this customer (outside the first 20,000 "
                 "customers precomputed in Phase 8 — in production this would run on-demand).")
    else:
        product_ids = row.iloc[0]["recommended_products"].split(",")
        st.subheader("Top 5 Recommended Products")
        details = product_stats[product_stats["product_id"].isin(product_ids)].set_index("product_id")
        details = details.reindex(product_ids)
        cols = st.columns(5)
        for i, pid in enumerate(product_ids):
            with cols[i % 5]:
                st.markdown(f"**#{i+1}**")
                st.code(pid[:12] + "...", language=None)
                if pid in details.index:
                    st.write(f"Category: {details.loc[pid, 'category']}")
                    st.write(f"Avg price: R$ {details.loc[pid, 'avg_price']:.2f}")
                    st.write(f"Rating: {details.loc[pid, 'avg_rating']:.1f} ⭐")

    st.markdown("---")
    st.subheader("Recommendation Method Comparison (Phase 8 evaluation)")
    _, _, reco_eval = load_model_comparisons()
    st.dataframe(reco_eval, width="stretch")
    st.caption("Evaluated via leave-last-product-out on customers with 2+ distinct purchases from "
               "the top-3,000 product catalog — see Phase 8 notebook for full methodology and caveats.")

# ---------------------------------------------------------------------------
# PAGE 4 — Business Insights
# ---------------------------------------------------------------------------
elif page == "Business Insights":
    st.title("Business Insights")

    orders = load_orders()
    items = load_items()
    delivered = orders[orders["order_status"] == "delivered"]

    tab1, tab2, tab3, tab4 = st.tabs(["Sales Trends", "Category Performance", "Seller Performance", "Geography"])

    with tab1:
        monthly = (delivered.set_index("order_purchase_timestamp")
                   .resample("MS")["total_payment_value"].sum().reset_index())
        fig = px.area(monthly, x="order_purchase_timestamp", y="total_payment_value")
        st.plotly_chart(fig, width="stretch")

        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        wd = delivered["purchase_weekday"].value_counts().reindex(weekday_order).reset_index()
        wd.columns = ["weekday", "orders"]
        fig2 = px.bar(wd, x="weekday", y="orders", title="Orders by Weekday")
        st.plotly_chart(fig2, width="stretch")

    with tab2:
        item_orders = items.merge(orders[["order_id", "order_status"]], on="order_id")
        item_orders = item_orders[item_orders["order_status"] == "delivered"]
        cat_rev = (item_orders.groupby("product_category_name_english")["price"]
                   .sum().sort_values(ascending=False).head(15).reset_index())
        fig = px.bar(cat_rev, x="price", y="product_category_name_english", orientation="h",
                     title="Top 15 Categories by Revenue")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width="stretch")

    with tab3:
        seller_rev = (item_orders.groupby("seller_id")["price"].sum()
                      .sort_values(ascending=False).head(15).reset_index())
        fig = px.bar(seller_rev, x="price", y="seller_id", orientation="h",
                     title="Top 15 Sellers by Revenue")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width="stretch")

        st.metric("Overall late delivery rate", f"{delivered['is_late'].mean():.1%}")
        st.metric("Median delivery time", f"{delivered['delivery_days'].median():.0f} days")

    with tab4:
        state_rev = delivered.groupby("customer_state")["total_payment_value"].sum().sort_values(
            ascending=False).reset_index()
        fig = px.bar(state_rev, x="customer_state", y="total_payment_value",
                     title="Revenue by Customer State")
        st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# PAGE 5 — Explainability
# ---------------------------------------------------------------------------
elif page == "Explainability":
    st.title("Explainability (SHAP)")

    clf_comp, clv_comp, _ = load_model_comparisons()

    st.subheader("Model Comparison Recap")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Purchase Prediction (Classification)**")
        st.dataframe(clf_comp.round(4), width="stretch")
    with col2:
        st.markdown("**CLV Prediction (Regression)**")
        st.dataframe(clv_comp.round(4), width="stretch")

    st.markdown("---")
    st.subheader("SHAP Visualizations")

    shap_images = {
        "Feature Importance (bar)": "shap_bar_purchase.png",
        "SHAP Summary — Purchase Prediction": "shap_summary_purchase.png",
        "SHAP Waterfall — Individual Customer": "shap_waterfall_purchase.png",
        "SHAP Force Plot — Individual Customer": "shap_force_purchase.png",
        "SHAP Summary — CLV Prediction": "shap_summary_clv.png",
    }
    for title, fname in shap_images.items():
        path = OUT + fname
        if os.path.exists(path):
            st.markdown(f"**{title}**")
            st.image(Image.open(path), width="stretch")
        else:
            st.info(f"{title}: image not found ({fname}). Run 09_explainability.ipynb first.")

    if os.path.exists(OUT + "shap_top_features_summary.csv"):
        st.markdown("---")
        st.subheader("Top Features Summary")
        st.dataframe(pd.read_csv(OUT + "shap_top_features_summary.csv"), width="stretch")
