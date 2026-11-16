import pandas as pd
import pytest

from src.train import chronological_split


def test_chronological_split_keeps_label_horizon_embargoes():
    dates = pd.date_range("2010-01-01", periods=24, freq="MS")
    frame = pd.DataFrame({
        "snapshot_date": dates,
        "customer_id": range(len(dates)),
        "repeat_purchase_90d": [0, 1] * 12,
    })

    train, validation, test = chronological_split(frame, embargo_days=90)

    train_end = train.snapshot_date.max()
    validation_start = validation.snapshot_date.min()
    validation_end = validation.snapshot_date.max()
    test_start = test.snapshot_date.min()

    assert validation_start >= train_end + pd.Timedelta(days=90)
    assert test_start >= validation_end + pd.Timedelta(days=90)
    assert train.snapshot_date.max() < validation.snapshot_date.min()
    assert validation.snapshot_date.max() < test.snapshot_date.min()


def test_chronological_split_rejects_too_few_dates():
    frame = pd.DataFrame({
        "snapshot_date": pd.date_range("2010-01-01", periods=5, freq="MS"),
        "repeat_purchase_90d": [0, 1, 0, 1, 0],
    })
    with pytest.raises(ValueError, match="at least 8"):
        chronological_split(frame)
