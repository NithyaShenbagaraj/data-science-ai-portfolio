import numpy as np
import pytest

from src.forecasting import seasonal_naive, naive_forecast
from src.metrics import mae, rmse, wape, forecast_bias


def test_seasonal_naive_repeats_last_season():
    assert seasonal_naive(list(range(1, 13)), 3, 12).tolist() == [1.0, 2.0, 3.0]


def test_daily_naive_repeats_last_week():
    assert naive_forecast(list(range(1, 15)), 9, 7).tolist() == [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 8.0, 9.0]


def test_metrics_are_finite_for_nonzero_actuals():
    actual = np.array([100.0, 120.0, 80.0])
    predicted = np.array([110.0, 115.0, 90.0])
    assert mae(actual, predicted) == pytest.approx(25 / 3)
    assert rmse(actual, predicted) > 0
    assert wape(actual, predicted) >= 0
    assert np.isfinite(forecast_bias(actual, predicted))
