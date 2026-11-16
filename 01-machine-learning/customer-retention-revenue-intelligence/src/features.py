"""Create monthly snapshots with past-only features and a future purchase label."""
from pathlib import Path
import pandas as pd
from src.data import DB_FILE, ROOT

FEATURES_FILE = ROOT / "data" / "processed" / "customer_snapshots.parquet"
LOOKBACK_DAYS = 180
HORIZON_DAYS = 90
NUMERIC_FEATURES = [
    "recency_days", "invoice_count_180d", "monetary_180d",
    "avg_invoice_value_180d", "units_180d", "distinct_products_180d",
    "active_days_180d",
]
CATEGORICAL_FEATURES = ["country"]
TARGET = "repeat_purchase_90d"


def build_customer_snapshots(transactions):
    sales = transactions.loc[
        transactions["is_valid_purchase"] & transactions["customer_id"].notna()
    ].copy()
    sales["customer_id"] = sales["customer_id"].astype("int64")
    sales["invoice_date"] = pd.to_datetime(sales["invoice_date"])
    sales["invoice_day"] = sales["invoice_date"].dt.normalize()
    if sales.empty:
        raise ValueError("No identified valid purchase rows available.")
    max_date = sales["invoice_date"].max().normalize()
    first_date = sales["invoice_date"].min().normalize()
    last_cutoff = max_date - pd.Timedelta(days=HORIZON_DAYS)
    cutoffs = pd.date_range(first_date + pd.Timedelta(days=LOOKBACK_DAYS), last_cutoff, freq="MS")
    snapshots = []

    for cutoff in cutoffs:
        history = sales[
            (sales["invoice_date"] < cutoff)
            & (sales["invoice_date"] >= cutoff - pd.Timedelta(days=LOOKBACK_DAYS))
        ]
        future = sales[
            (sales["invoice_date"] >= cutoff)
            & (sales["invoice_date"] < cutoff + pd.Timedelta(days=HORIZON_DAYS))
        ]
        if history.empty:
            continue
        grouped = history.groupby("customer_id", observed=True)
        features = grouped.agg(
            last_purchase=("invoice_date", "max"),
            invoice_count_180d=("invoice_no", "nunique"),
            monetary_180d=("line_revenue", "sum"),
            units_180d=("quantity", "sum"),
            distinct_products_180d=("stock_code", "nunique"),
            active_days_180d=("invoice_day", "nunique"),
            country=("country", lambda x: x.dropna().mode().iloc[0] if not x.dropna().mode().empty else "Unknown"),
        ).reset_index()
        invoice_values = (
            history.groupby(["customer_id", "invoice_no"], observed=True)["line_revenue"]
            .sum().groupby("customer_id").mean()
            .rename("avg_invoice_value_180d").reset_index()
        )
        features = features.merge(invoice_values, on="customer_id", how="left")
        features["recency_days"] = (cutoff - features["last_purchase"].dt.normalize()).dt.days
        future_customers = set(future["customer_id"].astype("int64").unique())
        features[TARGET] = features["customer_id"].isin(future_customers).astype("int8")
        features["snapshot_date"] = cutoff
        snapshots.append(features.drop(columns=["last_purchase"]))

    if not snapshots:
        raise ValueError("No complete monthly snapshots could be built.")
    result = pd.concat(snapshots, ignore_index=True)
    result["country"] = result["country"].fillna("Unknown").astype(str)
    return result.sort_values(["snapshot_date", "customer_id"]).reset_index(drop=True)


def main():
    if not DB_FILE.exists():
        raise FileNotFoundError("Run python -m src.data first.")
    import duckdb
    with duckdb.connect(str(DB_FILE), read_only=True) as con:
        transactions = con.execute("SELECT * FROM transactions").fetchdf()
    snapshots = build_customer_snapshots(transactions)
    FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    snapshots.to_parquet(FEATURES_FILE, index=False)
    print(f"Created {len(snapshots):,} snapshots at {FEATURES_FILE}")
    print(snapshots.groupby("snapshot_date")[TARGET].agg(["count", "mean"]).to_string())


if __name__ == "__main__":
    main()
