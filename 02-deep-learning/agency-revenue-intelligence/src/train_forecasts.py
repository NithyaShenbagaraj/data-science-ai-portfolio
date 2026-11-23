"""Train and compare the focused forecasting models on chronological splits."""
from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.data_access import load_forecasting_targets, resolve_data_dir
from src.features import split_by_time
from src.forecasting import (
    metric_row, seasonal_naive, naive_forecast, mean_forecast,
    sarima_forecast, xgb_recursive_forecast, lstm_recursive_forecast,
)


def evaluate_series(frame, date_col, target_col, frequency, output_dir):
    frame = frame[[date_col, target_col]].copy().sort_values(date_col).reset_index(drop=True)
    if frequency == "monthly":
        n = len(frame)
        train_end, val_end = 24, 30
        if n != 36:
            raise ValueError(f"Expected 36 agency monthly observations; found {n}")
        lags = (1, 2, 3, 6, 12)
        baseline = lambda h, k: seasonal_naive(h, k, 12)
    else:
        train, validation, test = split_by_time(frame, date_col)
        train_end = len(train)
        val_end = train_end + len(validation)
        lags = (1, 7, 14, 28)
        baseline = lambda h, k: naive_forecast(h, k, 7)

    train = frame.iloc[:train_end]
    validation = frame.iloc[train_end:val_end]
    test = frame.iloc[val_end:]
    if validation.empty or test.empty:
        raise ValueError("Chronological validation and test partitions must be non-empty")

    val_history = train[target_col].to_numpy(dtype=float)
    val_actual = validation[target_col].to_numpy(dtype=float)
    val_dates = validation[date_col]
    candidates = {}
    candidates["seasonal_naive" if frequency == "monthly" else "naive"] = baseline(val_history, len(validation))
    if frequency == "daily":
        candidates["mean_90"] = mean_forecast(
            val_history,
            len(validation),
            window=90,
        )
    if frequency == "monthly":
        try:
            candidates["sarima"] = sarima_forecast(val_history, len(validation))
        except Exception as exc:
            print(f"SARIMA validation failed: {exc}")
    try:
        pred, _ = xgb_recursive_forecast(train, val_dates, date_col=date_col, target_col=target_col, lags=lags)
        candidates["xgboost"] = pred
    except Exception as exc:
        print(f"XGBoost validation failed: {exc}")
    try:
        pred, _ = lstm_recursive_forecast(val_history, len(validation), lookback=min(12 if frequency == "monthly" else 28, len(val_history)-1))
        candidates["lstm"] = pred
    except Exception as exc:
        print(f"LSTM validation failed: {exc}")

    val_metrics = pd.DataFrame([metric_row(name, val_actual, pred) for name, pred in candidates.items()])
    selected = val_metrics.sort_values("wape").iloc[0]["model"]
    print(f"{frequency} selected on validation WAPE: {selected}")

    # Refit the selected family on train + validation, then score once on test.
    history = frame.iloc[:val_end]
    test_dates = test[date_col]
    history_values = history[target_col].to_numpy(dtype=float)
    if selected in ("seasonal_naive", "naive"):
        test_pred = baseline(history_values, len(test))
    elif selected == "mean_90":
        test_pred = mean_forecast(
            history_values,
            len(test),
            window=90,
        )
    elif selected == "sarima":
        test_pred = sarima_forecast(history_values, len(test))
    elif selected == "xgboost":
        test_pred, fitted = xgb_recursive_forecast(history, test_dates, date_col=date_col, target_col=target_col, lags=lags)
        joblib.dump(fitted, output_dir / f"{frequency}_xgboost.joblib")
    else:
        test_pred, fitted = lstm_recursive_forecast(history_values, len(test), lookback=min(12 if frequency == "monthly" else 28, len(history_values)-1))
        torch_path = output_dir / f"{frequency}_lstm_state.pt"
        import torch
        torch.save(fitted.state_dict(), torch_path)
    test_metrics = pd.DataFrame([metric_row(selected, test[target_col], test_pred)])
    predictions = pd.DataFrame({
        "date": test[date_col].to_numpy(),
        "actual": test[target_col].to_numpy(dtype=float),
        "predicted": test_pred,
        "target": frequency,
        "selected_model": selected,
        "partition": "test",
    })
    return val_metrics.assign(target=frequency), test_metrics.assign(target=frequency), predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default="reports/generated")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    agency, campaign = load_forecasting_targets(args.data_dir)
    monthly = evaluate_series(agency, "month", "recognized_agency_revenue", "monthly", output_dir)
    daily = evaluate_series(campaign, "date", "attributed_campaign_revenue", "daily", output_dir)
    validation = pd.concat([monthly[0], daily[0]], ignore_index=True)
    test = pd.concat([monthly[1], daily[1]], ignore_index=True)
    predictions = pd.concat([monthly[2], daily[2]], ignore_index=True)
    validation.to_csv(output_dir / "validation_metrics.csv", index=False)
    test.to_csv(output_dir / "test_metrics.csv", index=False)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    print("\nValidation metrics\n", validation.to_string(index=False))
    print("\nHeld-out test metrics\n", test.to_string(index=False))


if __name__ == "__main__":
    main()
