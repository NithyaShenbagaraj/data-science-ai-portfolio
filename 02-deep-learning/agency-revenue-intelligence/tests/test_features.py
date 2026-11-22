import numpy as np
import pandas as pd
import pytest

from src.features import make_lag_features, split_by_time


def test_lag_features_do_not_use_current_target():
    frame = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=12, freq="D"),
        "revenue": np.arange(12, dtype=float),
    })
    features = make_lag_features(frame, date_col="date", target_col="revenue", lags=(1, 3), rolling_windows=(3,))
    first = features.iloc[0]
    assert first["revenue"] == 3
    assert first["lag_1"] == 2
    assert first["lag_3"] == 0
    assert first["rolling_mean_3"] == pytest.approx(1.0)


def test_split_by_time_keeps_all_rows_for_a_date_together():
    frame = pd.DataFrame({
        "date": list(pd.date_range("2024-01-01", periods=10).repeat(2)),
        "value": range(20),
    })
    train, validation, test = split_by_time(frame, "date")
    assert set(train.date).isdisjoint(set(validation.date))
    assert set(train.date).isdisjoint(set(test.date))
    assert set(validation.date).isdisjoint(set(test.date))
    assert len(train) + len(validation) + len(test) == len(frame)


def test_invalid_split_fractions_raise():
    with pytest.raises(ValueError):
        split_by_time(pd.DataFrame({"date": pd.date_range("2024-01-01", periods=10)}), "date", 0.8, 0.2)
