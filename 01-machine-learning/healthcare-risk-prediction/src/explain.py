"""Generate SHAP explanations for the trained classification pipeline."""
from __future__ import annotations

import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
import joblib

from src.data import PROJECT_ROOT, load_dataset

MODEL_PATH = PROJECT_ROOT / "models" / "healthcare_risk_pipeline.joblib"
REPORT_DIR = PROJECT_ROOT / "reports"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Train the model first: python -m src.train")

    bundle = joblib.load(MODEL_PATH)
    pipeline = bundle["pipeline"]
    X, y = load_dataset()
    X_train, X_test, _, _ = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # Explain a small representative sample to keep permutation explanations tractable.
    background = X_train.sample(min(30, len(X_train)), random_state=42)
    examples = X_test.sample(min(30, len(X_test)), random_state=7)
    explainer = shap.Explainer(
        pipeline.predict_proba,
        background,
        algorithm="permutation",
    )
    shap_values = explainer(examples, max_evals=2 * len(X.columns) + 1)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    positive_class_values = shap_values[..., 1]
    shap.plots.beeswarm(positive_class_values, max_display=13, show=False)
    plt.gcf().set_size_inches(9, 6)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()

    print(f"Saved SHAP summary to {REPORT_DIR / 'shap_summary.png'}")
    print("Interpretation note: SHAP explains model output associations, not causality.")


if __name__ == "__main__":
    main()
