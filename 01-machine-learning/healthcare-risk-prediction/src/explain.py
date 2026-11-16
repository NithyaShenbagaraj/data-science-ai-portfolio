
"""Generate SHAP explanations for the trained classification pipeline."""

from __future__ import annotations

import matplotlib.pyplot as plt
import shap
import joblib

from sklearn.model_selection import train_test_split

from src.data import PROJECT_ROOT, load_dataset

MODEL_PATH = PROJECT_ROOT / "models" / "healthcare_risk_pipeline.joblib"
REPORT_DIR = PROJECT_ROOT / "reports"

BACKGROUND_SIZE = 30
EXPLANATION_SIZE = 30
RANDOM_STATE = 42
SHAP_EVAL_MULTIPLIER = 10


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Train the model first: python -m src.train"
        )

    bundle = joblib.load(MODEL_PATH)
    pipeline = bundle["pipeline"]

    X, y = load_dataset()

    X_train, X_test, _, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    background = X_train.sample(
        min(BACKGROUND_SIZE, len(X_train)),
        random_state=RANDOM_STATE,
    )

    examples = X_test.sample(
        min(EXPLANATION_SIZE, len(X_test)),
        random_state=7,
    )

    explainer = shap.Explainer(
        pipeline.predict_proba,
        background,
        algorithm="permutation",
    )

    max_evals = (
        SHAP_EVAL_MULTIPLIER * (2 * len(X.columns) + 1)
    )

    print(f"Explaining {len(examples)} observations...")
    print(f"Features: {len(X.columns)}")
    print(f"SHAP evaluation budget: {max_evals}")

    shap_values = explainer(
        examples,
        max_evals=max_evals,
    )

    # Explain positive-class probability (class 1).
    positive_class_values = shap_values[..., 1]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    shap.plots.beeswarm(
        positive_class_values,
        max_display=13,
        show=False,
    )

    plt.gcf().set_size_inches(10, 7)
    plt.tight_layout()

    output_path = REPORT_DIR / "shap_summary.png"

    plt.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved SHAP summary to {output_path}")
    print(
        "Interpretation note: SHAP explains model output "
        "contributions, not causality."
    )


if __name__ == "__main__":
    main()
