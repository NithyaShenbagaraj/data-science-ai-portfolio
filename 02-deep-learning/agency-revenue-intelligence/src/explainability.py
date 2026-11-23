"""SHAP explanations for fitted XGBoost tree models."""
from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_shap_values(model, feature_frame, max_rows=500):
    """Calculate SHAP values for a representative feature sample."""
    if feature_frame.empty:
        raise ValueError("Feature frame must not be empty")

    if max_rows < 1:
        raise ValueError("max_rows must be positive")

    sample = feature_frame.tail(max_rows).copy()

    import shap

    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(sample))

    if values.ndim == 3:
        values = values.mean(axis=0)

    if values.shape != sample.shape:
        raise ValueError(
            f"Unexpected SHAP shape {values.shape}; "
            f"expected {sample.shape}"
        )

    if not np.isfinite(values).all():
        raise ValueError("SHAP values contain non-finite values")

    shap_frame = pd.DataFrame(
        values,
        columns=sample.columns,
        index=sample.index,
    )

    return sample, shap_frame, explainer


def explain_tree_model(model, feature_frame, max_rows=500):
    """Return mean absolute SHAP values ranked by feature."""
    sample, shap_frame, _ = calculate_shap_values(
        model,
        feature_frame,
        max_rows=max_rows,
    )

    importance = (
        shap_frame.abs()
        .mean(axis=0)
        .rename("mean_abs_shap")
        .rename_axis("feature")
        .reset_index()
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    return importance


def explain_prediction(model, feature_row, explainer=None):
    """Return feature-level SHAP contributions for one observation."""
    if len(feature_row) != 1:
        raise ValueError("feature_row must contain exactly one observation")

    if explainer is None:
        import shap

        explainer = shap.TreeExplainer(model)

    explanation = explainer(feature_row)

    values = np.asarray(explanation.values).reshape(-1)

    if len(values) != feature_row.shape[1]:
        raise ValueError("SHAP output does not match the feature count")

    result = pd.DataFrame({
        "feature": feature_row.columns,
        "feature_value": feature_row.iloc[0].to_numpy(),
        "shap_value": values,
        "abs_shap_value": np.abs(values),
    })

    return result.sort_values(
        "abs_shap_value",
        ascending=False,
    ).reset_index(drop=True)
