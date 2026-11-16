import math

import numpy as np
import pytest

from src.metrics import forecast_bias, mae, rmse, wape


def test_error_metrics_for_perfect_forecast():
    actual = [10, 20, 30]
    predicted = [10, 20, 30]
    assert mae(actual, predicted) == 0
    assert rmse(actual, predicted) == 0
    assert wape(actual, predicted) == 0
    assert forecast_bias(actual, predicted) == 0


def test_metrics_for_non_perfect_forecast():
    assert mae([10, 20], [12, 18]) == pytest.approx(2)
    assert rmse([10, 20], [12, 18]) == pytest.approx(2)
    assert wape([10, 20], [12, 18]) == pytest.approx(4 / 30)


def test_wape_is_nan_when_actual_total_is_zero():
    assert math.isnan(wape([0, 0], [1, 2]))


def test_mismatched_shapes_raise_error():
    with pytest.raises(ValueError):
        mae(np.array([1, 2]), np.array([1]))
