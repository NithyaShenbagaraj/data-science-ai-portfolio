import numpy as np
import pytest

from src.forecasting import seasonal_naive, naive_forecast, mean_forecast
from src.metrics import mae, rmse, wape, forecast_bias


def test_seasonal_naive_repeats_last_season():
    assert seasonal_naive(list(range(1, 13)), 3, 12).tolist() == [1.0, 2.0, 3.0]


def test_daily_naive_repeats_last_week():
    assert naive_forecast(list(range(1, 15)), 9, 7).tolist() == [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 8.0, 9.0]

def test_mean_forecast_uses_latest_90_observations():
    history = np.arange(1, 101, dtype=float)

    predictions = mean_forecast(history, horizon=5, window=90)

    expected = np.full(5, np.mean(history[-90:]))

    np.testing.assert_array_equal(predictions, expected)


def test_mean_forecast_uses_available_history_when_less_than_90():
    history = np.array([10.0, 20.0, 30.0])

    predictions = mean_forecast(history, horizon=3, window=90)

    np.testing.assert_array_equal(
        predictions,
        np.full(3, 20.0),
    )


def test_mean_forecast_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        mean_forecast([], horizon=5, window=90)

    with pytest.raises(ValueError):
        mean_forecast([10.0, 20.0], horizon=0, window=90)

def test_metrics_are_finite_for_nonzero_actuals():
    actual = np.array([100.0, 120.0, 80.0])
    predicted = np.array([110.0, 115.0, 90.0])
    assert mae(actual, predicted) == pytest.approx(25 / 3)
    assert rmse(actual, predicted) > 0
    assert wape(actual, predicted) >= 0
    assert np.isfinite(forecast_bias(actual, predicted))
