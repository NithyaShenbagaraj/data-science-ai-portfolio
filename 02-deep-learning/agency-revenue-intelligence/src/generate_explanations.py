"""Generate leakage-safe SHAP artifacts for campaign revenue forecasting."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from xgboost import XGBRegressor

from src.data_access import load_forecasting_targets
from src.features import make_lag_features
from src.explainability import calculate_shap_values, explain_tree_model


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "reports" / "explainability"

DATE_COL = "date"
TARGET_COL = "attributed_campaign_revenue"
TEST_START = pd.Timestamp("2026-07-14")
LAGS = (1, 7, 14, 28)
ROLLING_WINDOWS = (3, 7)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _, campaign = load_forecasting_targets()

    # Exclude the held-out test period entirely.
    history = campaign.loc[campaign[DATE_COL] < TEST_START].copy()
    if history.empty:
        raise ValueError("No pre-test campaign history is available")

    features = make_lag_features(
        history,
        date_col=DATE_COL,
        target_col=TARGET_COL,
        lags=LAGS,
        rolling_windows=ROLLING_WINDOWS,
    )

    X = features.drop(columns=[DATE_COL, TARGET_COL])
    y = features[TARGET_COL].astype(float)

    model = XGBRegressor(
        n_estimators=350,
        max_depth=3,
        learning_rate=0.035,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=4,
    )
    model.fit(X, y)

    # Explain the pre-test historical feature rows only.
    sample, shap_frame, explainer = calculate_shap_values(
        model, X, max_rows=len(X)
    )

    importance = explain_tree_model(model, X, max_rows=len(X))
    importance.to_csv(
        OUTPUT_DIR / "shap_feature_importance.csv",
        index=False,
    )

    # Keep dates and actual values aligned with the SHAP sample.
    observation_values = pd.concat(
        [
            features.loc[sample.index, [DATE_COL, TARGET_COL]],
            shap_frame.add_prefix("shap_"),
        ],
        axis=1,
    )
    observation_values.to_csv(
        OUTPUT_DIR / "shap_observation_values.csv",
        index=False,
    )

    # Beeswarm plot: global distribution and direction of contributions.
    explanation = shap.Explanation(
        values=shap_frame.to_numpy(),
        data=sample.to_numpy(),
        feature_names=sample.columns.tolist(),
    )
    shap.plots.beeswarm(explanation, max_display=16, show=False)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "shap_summary.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close()

    print("SHAP artifact generation completed.")
    print("History rows:", len(history))
    print("History date range:",
          history[DATE_COL].min().date(), "to", history[DATE_COL].max().date())
    print("Feature rows:", len(features))
    print("Feature count:", X.shape[1])
    print("Test data used: No")
    print("\nTop 10 features:")
    print(importance.head(10).to_string(index=False))
    print("\nGenerated files:")
    for path in sorted(OUTPUT_DIR.iterdir()):
        print(path.relative_to(PROJECT_DIR))


if __name__ == "__main__":
    main()
