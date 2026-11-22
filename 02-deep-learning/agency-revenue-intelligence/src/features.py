"""Leakage-safe calendar, lag and rolling features for time-series models."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_lag_features(
    frame: pd.DataFrame,
    *,
    date_col: str,
    target_col: str,
    lags: tuple[int, ...],
    rolling_windows: tuple[int, ...] = (3, 7),
) -> pd.DataFrame:
    """Create one-step-ahead features; rolling statistics are shifted by one."""
    data = frame[[date_col, target_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(date_col).reset_index(drop=True)
    dates = data[date_col]
    target = pd.to_numeric(data[target_col], errors="raise")
    out = pd.DataFrame({date_col: dates, target_col: target})
    out["month"] = dates.dt.month
    out["quarter"] = dates.dt.quarter
    out["day_of_week"] = dates.dt.dayofweek
    out["day_of_year"] = dates.dt.dayofyear
    out["month_sin"] = np.sin(2 * np.pi * dates.dt.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * dates.dt.month / 12)
    out["dow_sin"] = np.sin(2 * np.pi * dates.dt.dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dates.dt.dayofweek / 7)
    for lag in lags:
        out[f"lag_{lag}"] = target.shift(lag)
    for window in rolling_windows:
        shifted = target.shift(1)
        out[f"rolling_mean_{window}"] = shifted.rolling(window).mean()
        out[f"rolling_std_{window}"] = shifted.rolling(window).std()
    return out.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def chronological_boundaries(n: int, train_fraction: float, validation_fraction: float):
    if n < 3 or not (0 < train_fraction < 1) or not (0 < validation_fraction < 1):
        raise ValueError("Need at least 3 observations and fractions between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must sum to less than 1")
    train_end = int(n * train_fraction)
    validation_end = train_end + int(n * validation_fraction)
    if train_end < 1 or validation_end <= train_end or validation_end >= n:
        raise ValueError("Fractions create an empty chronological split")
    return train_end, validation_end


def split_by_time(frame: pd.DataFrame, date_col: str, train_fraction=0.70, validation_fraction=0.15):
    """Split on sorted unique dates, keeping every row for a date in one partition."""
    dates = pd.Index(pd.to_datetime(frame[date_col]).drop_duplicates().sort_values())
    train_end, validation_end = chronological_boundaries(
        len(dates), train_fraction, validation_fraction
    )
    train_dates = set(dates[:train_end])
    validation_dates = set(dates[train_end:validation_end])
    test_dates = set(dates[validation_end:])
    date_values = pd.to_datetime(frame[date_col])
    return (
        frame.loc[date_values.isin(train_dates)].copy(),
        frame.loc[date_values.isin(validation_dates)].copy(),
        frame.loc[date_values.isin(test_dates)].copy(),
    )
