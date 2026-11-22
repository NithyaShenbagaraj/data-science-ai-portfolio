"""Optional SHAP explanations for fitted XGBoost tree models."""
from __future__ import annotations


def explain_tree_model(model, feature_frame, max_rows=500):
    """Return mean absolute SHAP values by feature; SHAP is loaded only on demand."""
    import shap
    sample = feature_frame.tail(max_rows)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(sample)
    import numpy as np
    import pandas as pd
    matrix = np.asarray(values)
    if matrix.ndim == 3:
        matrix = matrix.mean(axis=0)
    importance = np.abs(matrix).mean(axis=0)
    return (
        pd.DataFrame({"feature": sample.columns, "mean_abs_shap": importance})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
