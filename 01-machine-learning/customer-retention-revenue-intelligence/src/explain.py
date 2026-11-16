"""Generate SHAP global feature explanations for tree or linear models."""
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
from src.data import ROOT
from src.features import FEATURES_FILE, NUMERIC_FEATURES, CATEGORICAL_FEATURES

MODEL = ROOT / "models" / "repeat_purchase_pipeline.joblib"
OUT = ROOT / "reports"


def main():
    if not MODEL.exists() or not FEATURES_FILE.exists():
        raise FileNotFoundError("Run python -m src.train first.")
    model = joblib.load(MODEL)
    data = pd.read_parquet(FEATURES_FILE)
    test = data[data.snapshot_date >= "2011-07-01"]
    columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    sample = test[columns].sample(n=min(500, len(test)), random_state=42)
    transformed = model.named_steps["preprocess"].transform(sample)
    # OneHotEncoder may return a SciPy sparse matrix. SHAP TreeExplainer
    # expects a dense numeric array for this scikit-learn RandomForest model.
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    names = model.named_steps["preprocess"].get_feature_names_out()
    classifier = model.named_steps["model"]

    # Explain the fitted classifier using the already-transformed features.
    explainer = shap.TreeExplainer(classifier)
    explanation = explainer.shap_values(transformed)
    # SHAP versions differ: binary classifiers may return a list (one array
    # per class) or a 3-D array. Keep the positive-class explanation.
    if isinstance(explanation, list):
        values = explanation[1] if len(explanation) > 1 else explanation[0]
    else:
        values = explanation
        if getattr(values, "ndim", 0) == 3:
            values = values[:, :, 1]

    OUT.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(values, transformed, feature_names=names, show=False, max_display=18)
    plt.tight_layout()
    plt.savefig(OUT / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()
    importance = pd.DataFrame({
        "feature": names,
        "mean_abs_shap": abs(values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(OUT / "shap_feature_importance.csv", index=False)
    print(importance.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
