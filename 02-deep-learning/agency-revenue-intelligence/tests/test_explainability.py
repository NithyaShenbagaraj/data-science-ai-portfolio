import numpy as np
import pandas as pd
import pytest
from xgboost import XGBRegressor

from src.explainability import (
    calculate_shap_values,
    explain_tree_model,
    explain_prediction,
)


@pytest.fixture
def fitted_model_and_features():
    rng = np.random.default_rng(42)

    X = pd.DataFrame({
        "feature_a": rng.normal(size=40),
        "feature_b": rng.normal(size=40),
        "feature_c": rng.normal(size=40),
    })

    y = 3 * X["feature_a"] - 2 * X["feature_b"] + rng.normal(
        scale=0.1, size=40
    )

    model = XGBRegressor(
        n_estimators=10,
        max_depth=2,
        learning_rate=0.1,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
    )
    model.fit(X.iloc[:30], y.iloc[:30])

    return model, X.iloc[30:]


def test_calculate_shap_values(fitted_model_and_features):
    model, X = fitted_model_and_features

    sample, shap_frame, _ = calculate_shap_values(model, X)

    assert sample.shape == (10, 3)
    assert shap_frame.shape == (10, 3)
    assert list(shap_frame.columns) == list(X.columns)
    assert np.isfinite(shap_frame.to_numpy()).all()


def test_explain_tree_model_returns_sorted_importance(
    fitted_model_and_features,
):
    model, X = fitted_model_and_features

    importance = explain_tree_model(model, X)

    assert len(importance) == 3
    assert list(importance.columns) == ["feature", "mean_abs_shap"]
    assert importance["mean_abs_shap"].is_monotonic_decreasing
    assert (importance["mean_abs_shap"] >= 0).all()


def test_explain_prediction_returns_feature_contributions(
    fitted_model_and_features,
):
    model, X = fitted_model_and_features

    local = explain_prediction(model, X.iloc[[0]])

    assert len(local) == 3
    assert list(local.columns) == [
        "feature",
        "feature_value",
        "shap_value",
        "abs_shap_value",
    ]
    assert local["abs_shap_value"].is_monotonic_decreasing
    assert np.isfinite(local["shap_value"]).all()


def test_calculate_shap_values_rejects_empty_frame(
    fitted_model_and_features,
):
    model, X = fitted_model_and_features

    with pytest.raises(ValueError, match="must not be empty"):
        calculate_shap_values(model, X.iloc[:0])


def test_calculate_shap_values_rejects_invalid_max_rows(
    fitted_model_and_features,
):
    model, X = fitted_model_and_features

    with pytest.raises(ValueError, match="must be positive"):
        calculate_shap_values(model, X, max_rows=0)


def test_explain_prediction_requires_one_observation(
    fitted_model_and_features,
):
    model, X = fitted_model_and_features

    with pytest.raises(ValueError, match="exactly one"):
        explain_prediction(model, X.iloc[:2])
