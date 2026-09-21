"""
Phase 4 - Feature Engineering
Intelligent E-Commerce Customer Analytics Platform (Olist dataset)

Builds the final model-ready customer feature table on top of the Phase 1-2
outputs. Adds:
  - RFM quintile scores + combined RFM segment label (rule-based, pre-clustering)
  - Purchase cadence features (avg / std days between orders)
  - Product & seller diversity, favorite category
  - Review sentiment features
  - Geographic region grouping (Brazil's 5 macro-regions)
  - Price sensitivity features
  - Engagement ratios (order frequency rate, monetary per order, etc.)

This output (`customer_features.csv`) is the single dataset consumed by every
downstream modeling notebook (segmentation, purchase prediction, CLV, etc.).

Run:
    python 04_feature_engineering.py
Inputs:
    ./outputs/customer_master_clean.csv
    ./outputs/orders_master_clean.csv
    ./outputs/items_enriched.csv
Outputs:
    ./outputs/customer_features.csv
"""

import pandas as pd
import numpy as np

OUT = "outputs/"

STATE_TO_REGION = {
    # North
    "AC": "North", "AP": "North", "AM": "North", "PA": "North", "RO": "North", "RR": "North", "TO": "North",
    # Northeast
    "AL": "Northeast", "BA": "Northeast", "CE": "Northeast", "MA": "Northeast", "PB": "Northeast",
    "PE": "Northeast", "PI": "Northeast", "RN": "Northeast", "SE": "Northeast",
    # Central-West
    "DF": "Central-West", "GO": "Central-West", "MT": "Central-West", "MS": "Central-West",
    # Southeast
    "ES": "Southeast", "MG": "Southeast", "RJ": "Southeast", "SP": "Southeast",
    # South
    "PR": "South", "RS": "South", "SC": "South",
}


def add_rfm_scores(cust):
    """
    Quintile-based RFM scoring (1=worst, 5=best per dimension).
    Recency is inverted (lower days = better = higher score).
    """
    cust["R_score"] = pd.qcut(cust["recency_days"], 5, labels=[5, 4, 3, 2, 1]).astype(int)

    # frequency is heavily right-skewed / mostly 1s -> rank-based qcut with duplicates dropped
    cust["F_score"] = pd.qcut(cust["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    cust["M_score"] = pd.qcut(cust["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)

    cust["RFM_score"] = cust["R_score"] + cust["F_score"] + cust["M_score"]

    def label(row):
        if row["RFM_score"] >= 13:
            return "Champions"
        elif row["RFM_score"] >= 10:
            return "Loyal"
        elif row["R_score"] >= 4 and row["F_score"] <= 2:
            return "New/Promising"
        elif row["R_score"] <= 2 and row["F_score"] >= 3:
            return "At Risk"
        elif row["RFM_score"] <= 6:
            return "Hibernating"
        else:
            return "Needs Attention"

    cust["rfm_segment_rule_based"] = cust.apply(label, axis=1)
    return cust


def add_purchase_cadence(cust, orders):
    """Average & std of days between consecutive orders, per customer (repeat buyers only)."""
    delivered = orders[orders["order_status"] == "delivered"].sort_values("order_purchase_timestamp")
    gaps = (delivered.groupby("customer_unique_id")["order_purchase_timestamp"]
            .apply(lambda s: s.sort_values().diff().dt.days.dropna()))
    gap_stats = gaps.groupby(level=0).agg(["mean", "std"]).rename(
        columns={"mean": "avg_days_between_orders", "std": "std_days_between_orders"})
    gap_stats.index.name = "customer_unique_id"
    cust = cust.merge(gap_stats.reset_index(), on="customer_unique_id", how="left")
    # single-order customers have no gap -> fill with a sentinel (their tenure, i.e. "no repeat yet")
    cust["avg_days_between_orders"] = cust["avg_days_between_orders"].fillna(cust["customer_tenure_days"])
    cust["std_days_between_orders"] = cust["std_days_between_orders"].fillna(0)
    return cust


def add_product_seller_diversity(cust, orders, items):
    """Favorite category, category diversity, seller diversity, avg item price paid."""
    delivered_orders = orders[orders["order_status"] == "delivered"][["order_id", "customer_unique_id"]]
    item_cust = items.merge(delivered_orders, on="order_id", how="inner")

    cat_counts = (item_cust.groupby(["customer_unique_id", "product_category_name_english"])
                  .size().reset_index(name="n"))
    favorite_cat = (cat_counts.sort_values("n", ascending=False)
                    .drop_duplicates("customer_unique_id")[["customer_unique_id", "product_category_name_english"]]
                    .rename(columns={"product_category_name_english": "favorite_category"}))

    diversity = item_cust.groupby("customer_unique_id").agg(
        category_diversity=("product_category_name_english", "nunique"),
        seller_diversity=("seller_id", "nunique"),
        avg_price_paid=("price", "mean"),
        price_std=("price", "std"),
        avg_freight_value=("freight_value", "mean"),
    ).reset_index()

    cust = cust.merge(favorite_cat, on="customer_unique_id", how="left")
    cust = cust.merge(diversity, on="customer_unique_id", how="left")
    cust["favorite_category"] = cust["favorite_category"].fillna("unknown")
    cust["price_std"] = cust["price_std"].fillna(0)
    # Customers with zero delivered orders have no item-level history at all
    for col in ["category_diversity", "seller_diversity", "avg_price_paid", "avg_freight_value"]:
        cust[col] = cust[col].fillna(0)
    return cust


def add_review_sentiment_features(cust, orders):
    delivered = orders[orders["order_status"] == "delivered"]
    rev = delivered.groupby("customer_unique_id").agg(
        pct_low_reviews=("review_score", lambda s: (s <= 2).mean()),
        pct_high_reviews=("review_score", lambda s: (s >= 4).mean()),
    ).reset_index()
    cust = cust.merge(rev, on="customer_unique_id", how="left")
    cust["pct_low_reviews"] = cust["pct_low_reviews"].fillna(0)
    cust["pct_high_reviews"] = cust["pct_high_reviews"].fillna(0)
    return cust


def add_geography(cust):
    cust["customer_region"] = cust["customer_state"].map(STATE_TO_REGION).fillna("unknown")
    return cust


def add_engagement_ratios(cust):
    cust["order_frequency_rate"] = cust["frequency"] / cust["customer_tenure_days"].clip(lower=1) * 30
    cust["monetary_per_order"] = cust["monetary"] / cust["frequency"].clip(lower=1)
    cust["recency_to_tenure_ratio"] = cust["recency_days"] / cust["customer_tenure_days"].clip(lower=1)
    return cust


def main():
    cust = pd.read_csv(OUT + "customer_master_clean.csv", parse_dates=["first_purchase", "last_purchase"])
    orders = pd.read_csv(OUT + "orders_master_clean.csv", parse_dates=["order_purchase_timestamp"])
    items = pd.read_csv(OUT + "items_enriched.csv")

    print("Starting shape:", cust.shape)

    cust = add_rfm_scores(cust)
    print("After RFM scores:", cust.shape)

    cust = add_purchase_cadence(cust, orders)
    print("After purchase cadence:", cust.shape)

    cust = add_product_seller_diversity(cust, orders, items)
    print("After product/seller diversity:", cust.shape)

    cust = add_review_sentiment_features(cust, orders)
    print("After review sentiment:", cust.shape)

    cust = add_geography(cust)
    cust = add_engagement_ratios(cust)
    print("Final shape:", cust.shape)

    print("\nRFM segment distribution (rule-based, pre-clustering baseline):")
    print(cust["rfm_segment_rule_based"].value_counts())

    print("\nRegion distribution:")
    print(cust["customer_region"].value_counts())

    print("\nNull check:")
    nulls = cust.isna().sum()
    print(nulls[nulls > 0])

    print("\nFeature columns:", list(cust.columns))

    cust.to_csv(OUT + "customer_features.csv", index=False)
    print("\nSaved: customer_features.csv")


if __name__ == "__main__":
    main()
