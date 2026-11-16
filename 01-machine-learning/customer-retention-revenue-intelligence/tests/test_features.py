import pandas as pd
from src.data import clean_transactions
from src.features import build_customer_snapshots


def make_raw(rows):
    return pd.DataFrame(rows, columns=[
        "invoice_no", "stock_code", "description", "quantity",
        "invoice_date", "unit_price", "customer_id", "country",
    ])


def test_cleaner_flags_cancellations():
    raw = make_raw([
        ["100", "A", "item", 2, "2010-06-01", 5.0, 1, "UK"],
        ["C101", "A", "item", -1, "2010-06-02", 5.0, 1, "UK"],
    ])
    clean = clean_transactions(raw)
    cancel = clean.loc[clean.invoice_no == "C101"].iloc[0]
    assert bool(cancel.is_cancellation)
    assert not bool(cancel.is_valid_purchase)


def test_snapshot_uses_only_past_data_and_future_label():
    raw = make_raw([
        ["090", "Z", "old", 1, "2010-01-01", 1.0, 99, "UK"],
        ["200", "A", "item", 2, "2010-06-01", 5.0, 7, "UK"],
        ["201", "B", "item", 1, "2010-08-01", 10.0, 7, "UK"],
        ["202", "C", "item", 1, "2010-11-01", 4.0, 7, "UK"],
    ])
    snapshots = build_customer_snapshots(clean_transactions(raw))
    row = snapshots.loc[
        (snapshots.snapshot_date == pd.Timestamp("2010-07-01"))
        & (snapshots.customer_id == 7)
    ].iloc[0]
    assert row.invoice_count_180d == 1
    assert row.monetary_180d == 10.0
    assert row.repeat_purchase_90d == 1


def test_snapshot_date_purchase_is_future_not_history():
    raw = make_raw([
        ["090", "Z", "old", 1, "2010-01-01", 1.0, 99, "UK"],
        ["300", "A", "item", 1, "2010-06-01", 3.0, 9, "UK"],
        ["301", "B", "item", 1, "2010-07-01", 4.0, 9, "UK"],
        ["302", "C", "item", 1, "2010-10-01", 2.0, 10, "UK"],
    ])
    snapshots = build_customer_snapshots(clean_transactions(raw))
    row = snapshots.loc[
        (snapshots.snapshot_date == pd.Timestamp("2010-07-01"))
        & (snapshots.customer_id == 9)
    ].iloc[0]
    assert row.invoice_count_180d == 1
    assert row.repeat_purchase_90d == 1
