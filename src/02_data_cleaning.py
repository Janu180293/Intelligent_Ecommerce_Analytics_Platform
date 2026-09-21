"""
Phase 2 - Data Cleaning & Quality Assurance
Intelligent E-Commerce Customer Analytics Platform (Olist dataset)

Takes the outputs of 01_data_integration.py and:
  - Handles missing values (context-aware, not blanket dropna/fillna(0))
  - Checks for duplicate keys
  - Flags (does not silently delete) statistical outliers in spend/behavior,
    since in a CLV/segmentation project the "outliers" are often your best customers
  - Validates against impossible values (negative prices, negative durations)

Run:
    python 02_data_cleaning.py
Inputs:
    ./outputs/orders_master.csv
    ./outputs/customer_master.csv
Outputs:
    ./outputs/orders_master_clean.csv
    ./outputs/customer_master_clean.csv
"""

import pandas as pd
import numpy as np

OUT = "outputs/"

DATE_COLS_ORDERS = [
    "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
    "order_delivered_customer_date", "order_estimated_delivery_date",
]


def iqr_bounds(series, k=3.0):
    """Wide IQR fence (k=3, vs the usual k=1.5) — deliberately conservative since
    this is a retail/CLV context where high spenders are meaningful, not noise."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def clean_customer_master(cust):
    print("customer_master nulls before cleaning:")
    print(cust.isna().sum()[cust.isna().sum() > 0])

    # No delivered order -> no observed payment method; explicit category, not a guess
    cust["preferred_payment_method"] = cust["preferred_payment_method"].fillna("unknown")
    # A handful of customers have no review/delivery observation; impute with the
    # population median rather than 0 (0 would look like "worst possible" which is wrong)
    cust["avg_review_rating"] = cust["avg_review_rating"].fillna(cust["avg_review_rating"].median())
    cust["avg_delivery_days"] = cust["avg_delivery_days"].fillna(cust["avg_delivery_days"].median())
    cust["avg_delivery_days"] = cust["avg_delivery_days"].clip(lower=0)

    n_dupes = cust.duplicated("customer_unique_id").sum()
    print(f"\nDuplicate customer_unique_id rows: {n_dupes}")
    cust = cust.drop_duplicates("customer_unique_id")

    print("\nOutlier scan (IQR, k=3 — flagged only, not removed):")
    for col in ["total_spending", "avg_order_value", "total_items_bought", "avg_delivery_days"]:
        lo, hi = iqr_bounds(cust[col])
        mask = (cust[col] < lo) | (cust[col] > hi)
        print(f"  {col:20s} bounds=({lo:.1f}, {hi:.1f})  n_outliers={mask.sum()} ({100*mask.mean():.2f}%)")

    for col in ["total_spending", "avg_order_value"]:
        lo, hi = iqr_bounds(cust[col])
        cust[f"{col}_outlier_flag"] = ((cust[col] < lo) | (cust[col] > hi)).astype(int)

    return cust


def clean_orders_master(om):
    print("\norders_master nulls before cleaning (many are structural — e.g. an order")
    print("that was never delivered legitimately has no delivery date):")
    print(om.isna().sum()[om.isna().sum() > 0])

    before = len(om)
    om = om.drop_duplicates("order_id")
    print(f"\nDropped {before - len(om)} duplicate order_id rows")

    n_negative_price = (om["order_items_price"] < 0).sum()
    n_negative_freight = (om["order_freight_value"] < 0).sum()
    print(f"Negative order_items_price rows: {n_negative_price}")
    print(f"Negative order_freight_value rows: {n_negative_freight}")
    # Guard clause — if any appear in a re-run on different data, do not silently keep them
    om = om[(om["order_items_price"].isna()) | (om["order_items_price"] >= 0)]
    om = om[(om["order_freight_value"].isna()) | (om["order_freight_value"] >= 0)]

    return om


def main():
    cust = pd.read_csv(OUT + "customer_master.csv", parse_dates=["first_purchase", "last_purchase"])
    om = pd.read_csv(OUT + "orders_master.csv", parse_dates=DATE_COLS_ORDERS)

    cust_clean = clean_customer_master(cust)
    om_clean = clean_orders_master(om)

    print("\nFinal shapes:")
    print("  customer_master_clean:", cust_clean.shape)
    print("  orders_master_clean:  ", om_clean.shape)

    cust_clean.to_csv(OUT + "customer_master_clean.csv", index=False)
    om_clean.to_csv(OUT + "orders_master_clean.csv", index=False)
    print("\nSaved: customer_master_clean.csv, orders_master_clean.csv")


if __name__ == "__main__":
    main()
