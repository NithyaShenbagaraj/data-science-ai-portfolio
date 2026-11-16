"""Forecasting metrics with explicit handling of zero-valued actuals."""

from __future__ import annotations

import numpy as np


def mae(y_true, y_pred) -> float:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.mean(np.abs(actual - predicted)))


def rmse(y_true, y_pred) -> float:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error; returns NaN if actual sum is zero."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    denominator = np.sum(np.abs(actual))
    if denominator == 0:
        return float("nan")
    return float(np.sum(np.abs(actual - predicted)) / denominator)


def forecast_bias(y_true, y_pred) -> float:
    """Signed bias as total forecast minus total actual, divided by total actual magnitude."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    denominator = np.sum(np.abs(actual))
    if denominator == 0:
        return float("nan")
    return float(np.sum(predicted - actual) / denominator)
