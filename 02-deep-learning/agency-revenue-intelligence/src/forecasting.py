"""Time-series baselines and supervised forecasting helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

from src.features import make_lag_features
from src.metrics import mae, rmse, wape, forecast_bias


def metric_row(model_name, actual, predicted):
    return {
        "model": model_name,
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "wape": wape(actual, predicted),
        "forecast_bias": forecast_bias(actual, predicted),
    }


def seasonal_naive(history, horizon: int, season_length: int = 12):
    values = np.asarray(history, dtype=float)
    if len(values) == 0 or horizon < 1:
        raise ValueError("History and horizon must be non-empty/positive")
    if len(values) < season_length:
        return np.repeat(values[-1], horizon)
    return np.resize(values[-season_length:], horizon)


def naive_forecast(history, horizon: int, lag: int = 7):
    values = np.asarray(history, dtype=float)
    if len(values) == 0 or horizon < 1:
        raise ValueError("History and horizon must be non-empty/positive")
    if len(values) < lag:
        return np.repeat(values[-1], horizon)
    return np.resize(values[-lag:], horizon)

def mean_forecast(history, horizon: int, window: int = 90):
    """Forecast using the mean of the most recent observations."""
    values = np.asarray(history, dtype=float)

    if len(values) == 0 or horizon < 1:
        raise ValueError("History and horizon must be non-empty/positive")

    recent_mean = values[-min(window, len(values)):].mean()

    return np.full(horizon, recent_mean)

def sarima_forecast(history, horizon: int, order=(1, 1, 1)):
    values = np.asarray(history, dtype=float)
    model = SARIMAX(
        values, order=order, trend="t",
        enforce_stationarity=False, enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    return np.maximum(0, np.asarray(fitted.forecast(horizon), dtype=float))


def _supervised(frame, date_col, target_col, lags, rolling):
    features = make_lag_features(
        frame, date_col=date_col, target_col=target_col,
        lags=lags, rolling_windows=rolling,
    )
    x = features.drop(columns=[date_col, target_col])
    y = features[target_col].astype(float)
    return features, x, y


def xgb_recursive_forecast(history_frame, future_dates, *, date_col, target_col, lags):
    """Recursive one-step XGBoost forecast; future target values are never features."""
    history = history_frame[[date_col, target_col]].copy().sort_values(date_col)
    future_dates = pd.to_datetime(pd.Series(future_dates))
    if len(history) <= max(lags):
        raise ValueError("Not enough history for the requested lag features")
    features, x, y = _supervised(
        history, date_col, target_col, tuple(lags),
        tuple(sorted(set([min(3, max(lags)), min(7, max(lags))]))),
    )
    model = XGBRegressor(
        n_estimators=350, max_depth=3, learning_rate=0.035,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=2.0,
        objective="reg:squarederror", random_state=42, n_jobs=4,
    )
    model.fit(x, y)
    values = history[target_col].astype(float).tolist()
    predictions = []
    for date in future_dates:
        row = {
            "month": date.month, "quarter": date.quarter,
            "day_of_week": date.dayofweek, "day_of_year": date.dayofyear,
            "month_sin": np.sin(2 * np.pi * date.month / 12),
            "month_cos": np.cos(2 * np.pi * date.month / 12),
            "dow_sin": np.sin(2 * np.pi * date.dayofweek / 7),
            "dow_cos": np.cos(2 * np.pi * date.dayofweek / 7),
        }
        for lag in lags:
            row[f"lag_{lag}"] = values[-lag]
        for window in sorted(set([min(3, max(lags)), min(7, max(lags))])):
            recent = np.asarray(values[-window:], dtype=float)
            row[f"rolling_mean_{window}"] = float(recent.mean())
            row[f"rolling_std_{window}"] = float(recent.std(ddof=1)) if len(recent) > 1 else 0.0
        row_frame = pd.DataFrame([row]).reindex(columns=x.columns, fill_value=0)
        prediction = max(0.0, float(model.predict(row_frame)[0]))
        predictions.append(prediction)
        values.append(prediction)
    return np.asarray(predictions), model


def lstm_recursive_forecast(history, horizon, lookback=12, epochs=60):
    """Small univariate PyTorch LSTM; returns non-negative recursive forecasts."""
    import torch
    from torch import nn
    from sklearn.preprocessing import MinMaxScaler

    # Avoid native threading conflicts in the macOS CPU environment.
    torch.set_num_threads(1)

    values = np.asarray(history, dtype=np.float32).reshape(-1, 1)
    if len(values) <= lookback:
        raise ValueError("History must be longer than LSTM lookback")
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values).astype(np.float32)
    x = np.stack([scaled[i-lookback:i] for i in range(lookback, len(scaled))])
    y = scaled[lookback:]
    torch.manual_seed(42)
    xt = torch.tensor(x)
    yt = torch.tensor(y)
    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, 32, batch_first=True)
            self.head = nn.Linear(32, 1)
        def forward(self, batch):
            output, _ = self.lstm(batch)
            return self.head(output[:, -1, :])
    model = Network()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        optimizer.step()
    window = scaled[-lookback:].copy()
    forecasts = []
    model.eval()
    with torch.no_grad():
        for _ in range(horizon):
            pred = model(torch.tensor(window.reshape(1, lookback, 1))).numpy()[0, 0]
            forecasts.append(pred)
            window = np.vstack([window[1:], [[pred]]])
    inverse = scaler.inverse_transform(np.asarray(forecasts).reshape(-1, 1)).ravel()
    return np.maximum(0, inverse), model
