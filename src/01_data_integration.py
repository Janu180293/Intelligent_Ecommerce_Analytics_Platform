"""
Phase 1 - Data Integration
Intelligent E-Commerce Customer Analytics Platform (Olist dataset)

Merges the 9 raw Olist tables into two analytical datasets:
  1. orders_master.csv    - one row per order, with customer/product/payment/review context
  2. customer_master.csv  - one row per unique customer, with RFM (Recency/Frequency/Monetary)
                             and behavioral aggregates

IMPORTANT OLIST QUIRK:
  `customer_id` is unique PER ORDER, not per person. The true, persistent customer
  identifier across repeat purchases is `customer_unique_id`. All customer-level
  aggregation in this project uses `customer_unique_id`.

Run:
    python 01_data_integration.py
Inputs:
    ./data/olist_*.csv, ./data/product_category_name_translation.csv
Outputs:
    ./outputs/items_enriched.csv
    ./outputs/orders_master.csv
    ./outputs/customer_master.csv
"""

import pandas as pd
import numpy as np

DATA = "data/"
OUT = "outputs/"

DATE_COLS_ORDERS = [
    "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
    "order_delivered_customer_date", "order_estimated_delivery_date",
]


def load_raw_tables():
    """Load all raw Olist CSVs with correct dtypes/date parsing."""
    tables = {
        "customers":   pd.read_csv(DATA + "olist_customers_dataset.csv"),
        "orders":      pd.read_csv(DATA + "olist_orders_dataset.csv", parse_dates=DATE_COLS_ORDERS),
        "order_items": pd.read_csv(DATA + "olist_order_items_dataset.csv", parse_dates=["shipping_limit_date"]),
        "payments":    pd.read_csv(DATA + "olist_order_payments_dataset.csv"),
        "reviews":     pd.read_csv(DATA + "olist_order_reviews_dataset.csv",
                                    parse_dates=["review_creation_date", "review_answer_timestamp"]),
        "products":    pd.read_csv(DATA + "olist_products_dataset.csv"),
        "sellers":     pd.read_csv(DATA + "olist_sellers_dataset.csv"),
        "geoloc":      pd.read_csv(DATA + "olist_geolocation_dataset.csv"),
        "cat_trans":   pd.read_csv(DATA + "product_category_name_translation.csv"),
    }
    print("Raw table shapes:")
    for name, df in tables.items():
        print(f"  {name:15s} {df.shape}")
    return tables


def build_items_enriched(tables):
    """Item-level table: order_items + product info (w/ English category) + seller info."""
    products = tables["products"].merge(tables["cat_trans"], on="product_category_name", how="left")
    products["product_category_name_english"] = (
        products["product_category_name_english"]
        .fillna(products["product_category_name"])
        .fillna("unknown")
    )
    items = tables["order_items"].merge(products, on="product_id", how="left")
    items = items.merge(tables["sellers"], on="seller_id", how="left")
    return items


def aggregate_to_order_level(tables, items):
    """Collapse items/payments/reviews (which can be many-rows-per-order) to one row per order_id."""
    pay_agg = tables["payments"].groupby("order_id").agg(
        total_payment_value=("payment_value", "sum"),
        n_payment_installments=("payment_installments", "max"),
        n_payment_methods=("payment_type", "nunique"),
    ).reset_index()
    # dominant payment type by value, used as the order's "primary" method
    pay_type_primary = (
        tables["payments"].sort_values("payment_value", ascending=False)
        .drop_duplicates("order_id")[["order_id", "payment_type"]]
        .rename(columns={"payment_type": "primary_payment_type"})
    )
    pay_agg = pay_agg.merge(pay_type_primary, on="order_id", how="left")

    rev_agg = tables["reviews"].groupby("order_id").agg(
        review_score=("review_score", "mean"),
        n_reviews=("review_id", "count"),
    ).reset_index()

    item_agg = items.groupby("order_id").agg(
        n_items=("order_item_id", "count"),
        n_distinct_products=("product_id", "nunique"),
        n_distinct_sellers=("seller_id", "nunique"),
        order_items_price=("price", "sum"),
        order_freight_value=("freight_value", "sum"),
        avg_item_price=("price", "mean"),
    ).reset_index()

    return item_agg, pay_agg, rev_agg


def build_orders_master(tables, item_agg, pay_agg, rev_agg):
    """Final order-level table with derived timing/delivery features."""
    om = tables["orders"].merge(tables["customers"], on="customer_id", how="left")
    om = om.merge(item_agg, on="order_id", how="left")
    om = om.merge(pay_agg, on="order_id", how="left")
    om = om.merge(rev_agg, on="order_id", how="left")

    om["delivery_days"] = (om["order_delivered_customer_date"] - om["order_purchase_timestamp"]).dt.days
    om["estimated_delivery_days"] = (om["order_estimated_delivery_date"] - om["order_purchase_timestamp"]).dt.days
    om["delivery_delay_days"] = (om["order_delivered_customer_date"] - om["order_estimated_delivery_date"]).dt.days
    om["is_late"] = om["delivery_delay_days"] > 0
    om["purchase_month"] = om["order_purchase_timestamp"].dt.month
    om["purchase_weekday"] = om["order_purchase_timestamp"].dt.day_name()
    om["purchase_hour"] = om["order_purchase_timestamp"].dt.hour
    om["purchase_year"] = om["order_purchase_timestamp"].dt.year
    return om


def build_customer_master(orders_master):
    """
    Customer-level RFM + behavioral aggregate table, keyed by customer_unique_id.
    Monetary/behavioral metrics are computed on 'delivered' orders only (an order that
    was canceled/unavailable never generated real revenue or a real delivery experience).
    Order *count* (frequency), however, is based on all orders placed.
    """
    snapshot_date = orders_master["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

    delivered = orders_master[orders_master["order_status"] == "delivered"]

    cust = orders_master.groupby("customer_unique_id").agg(
        total_orders=("order_id", "nunique"),
        first_purchase=("order_purchase_timestamp", "min"),
        last_purchase=("order_purchase_timestamp", "max"),
        customer_state=("customer_state", "first"),
        customer_city=("customer_city", "first"),
    ).reset_index()

    monetary = delivered.groupby("customer_unique_id").agg(
        total_spending=("total_payment_value", "sum"),
        avg_order_value=("total_payment_value", "mean"),
        total_items_bought=("n_items", "sum"),
        avg_basket_size=("n_items", "mean"),
        avg_review_rating=("review_score", "mean"),
        avg_delivery_days=("delivery_days", "mean"),
        late_delivery_rate=("is_late", "mean"),
        n_delivered_orders=("order_id", "nunique"),
    ).reset_index()

    cust = cust.merge(monetary, on="customer_unique_id", how="left")

    pref_pay = (
        delivered.groupby(["customer_unique_id", "primary_payment_type"]).size()
        .reset_index(name="cnt")
        .sort_values("cnt", ascending=False)
        .drop_duplicates("customer_unique_id")[["customer_unique_id", "primary_payment_type"]]
        .rename(columns={"primary_payment_type": "preferred_payment_method"})
    )
    cust = cust.merge(pref_pay, on="customer_unique_id", how="left")

    cust["recency_days"] = (snapshot_date - cust["last_purchase"]).dt.days
    cust["frequency"] = cust["total_orders"]
    cust["monetary"] = cust["total_spending"].fillna(0)
    cust["customer_tenure_days"] = (snapshot_date - cust["first_purchase"]).dt.days
    cust["days_since_last_purchase"] = cust["recency_days"]
    cust["is_repeat_customer"] = (cust["total_orders"] > 1).astype(int)

    fill_zero_cols = ["total_spending", "avg_order_value", "total_items_bought", "avg_basket_size",
                       "avg_review_rating", "avg_delivery_days", "late_delivery_rate", "n_delivered_orders"]
    cust[fill_zero_cols] = cust[fill_zero_cols].fillna(0)

    return cust, snapshot_date


def main():
    tables = load_raw_tables()

    print("\nOlist ID check — customer_id (per-order) vs customer_unique_id (per-person):")
    print("  unique customer_id       :", tables["customers"]["customer_id"].nunique())
    print("  unique customer_unique_id:", tables["customers"]["customer_unique_id"].nunique())

    items = build_items_enriched(tables)
    item_agg, pay_agg, rev_agg = aggregate_to_order_level(tables, items)
    orders_master = build_orders_master(tables, item_agg, pay_agg, rev_agg)
    customer_master, snapshot_date = build_customer_master(orders_master)

    print(f"\nSnapshot date used for recency calc: {snapshot_date}")
    print("orders_master shape:  ", orders_master.shape)
    print("customer_master shape:", customer_master.shape)

    items.to_csv(OUT + "items_enriched.csv", index=False)
    orders_master.to_csv(OUT + "orders_master.csv", index=False)
    customer_master.to_csv(OUT + "customer_master.csv", index=False)
    print("\nSaved: items_enriched.csv, orders_master.csv, customer_master.csv")


if __name__ == "__main__":
    main()
