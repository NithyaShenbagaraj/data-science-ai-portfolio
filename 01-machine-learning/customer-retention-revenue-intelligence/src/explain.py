
"""Generate SHAP global feature explanations for the trained model."""
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

from src.data import ROOT
from src.features import (
    FEATURES_FILE,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)

MODEL = ROOT / "models" / "repeat_purchase_pipeline.joblib"
OUT = ROOT / "reports"


def main():
    # Check that the trained model and feature data exist.
    if not MODEL.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL}. Run python -m src.train first."
        )

    if not FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"Feature data not found: {FEATURES_FILE}. "
            "Run feature engineering first."
        )

    # Load the trained pipeline and feature data.
    model = joblib.load(MODEL)
    data = pd.read_parquet(FEATURES_FILE)

    # Select the later-period observations for explanation.
    test = data[data["snapshot_date"] >= "2011-07-01"].copy()

    if test.empty:
        raise ValueError(
            "No observations found after 2011-07-01. "
            "Check the snapshot_date values in the feature dataset."
        )

    columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    sample = test[columns].sample(
        n=min(500, len(test)),
        random_state=42,
    )

    # Apply the preprocessing steps from the trained pipeline.
    preprocessor = model.named_steps["preprocess"]
    transformed = preprocessor.transform(sample)

    # SHAP TreeExplainer requires dense numeric input in this workflow.
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    names = preprocessor.get_feature_names_out()
    classifier = model.named_steps["model"]

    print("Classifier:", type(classifier).__name__)
    print("Transformed data shape:", transformed.shape)
    print("Feature names:", len(names))

    # Explain the Random Forest classifier directly.
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(transformed)

    # SHAP versions return different formats for binary classification.
    if isinstance(shap_values, list):
        values = (
            shap_values[1]
            if len(shap_values) > 1
            else shap_values[0]
        )
    else:
        values = shap_values

        if getattr(values, "ndim", 0) == 3:
            values = values[:, :, 1]

    # Confirm that the explanation dimensions match the input features.
    if values.ndim != 2:
        raise ValueError(
            f"Expected 2D SHAP values, received shape: {values.shape}"
        )

    if values.shape[1] != len(names):
        raise ValueError(
            f"SHAP feature count ({values.shape[1]}) does not match "
            f"preprocessed feature count ({len(names)})."
        )

    # Create the reports directory and save the summary plot.
    OUT.mkdir(parents=True, exist_ok=True)

    shap.summary_plot(
        values,
        transformed,
        feature_names=names,
        show=False,
        max_display=18,
    )

    plt.tight_layout()
    plt.savefig(
        OUT / "shap_summary.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close()

    # Calculate global feature importance.
    importance = pd.DataFrame({
        "feature": names,
        "mean_abs_shap": abs(values).mean(axis=0),
    }).sort_values(
        "mean_abs_shap",
        ascending=False,
    )

    importance.to_csv(
        OUT / "shap_feature_importance.csv",
        index=False,
    )

    print("\nTop 15 SHAP features:")
    print(importance.head(15).to_string(index=False))

    print("\nSHAP reports generated successfully:")
    print(OUT / "shap_summary.png")
    print(OUT / "shap_feature_importance.csv")


if __name__ == "__main__":
    main()
