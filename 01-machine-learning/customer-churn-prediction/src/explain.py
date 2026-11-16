"""Generate permutation importance and SHAP explanations."""
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from src.data import DEFAULT_DATA_PATH, PROJECT_ROOT, load_dataset
from src.features import split_features_target

MODEL_PATH = PROJECT_ROOT / "models" / "customer_churn_pipeline.joblib"
REPORT_DIR = PROJECT_ROOT / "reports"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Train the model first with: python -m src.train")
    X, y = split_features_target(load_dataset(DEFAULT_DATA_PATH))
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = joblib.load(MODEL_PATH)
    result = permutation_importance(
        model, X_test, y_test, scoring="roc_auc", n_repeats=8,
        random_state=42, n_jobs=-1
    )
    importance = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    importance.to_csv(REPORT_DIR / "permutation_importance.csv", index=False)
    background = X_test.sample(min(40, len(X_test)), random_state=42)
    explain_rows = X_test.sample(min(100, len(X_test)), random_state=42)
    explainer = shap.Explainer(model.predict_proba, background)
    values = explainer(explain_rows)
    shap.plots.beeswarm(values[:, :, 1], max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved explanations to {REPORT_DIR}")


if __name__ == "__main__":
    main()
