"""Train, compare and evaluate leakage-safe churn classifiers."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, ConfusionMatrixDisplay,
    f1_score, precision_score, recall_score, roc_auc_score, RocCurveDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.data import DEFAULT_DATA_PATH, PROJECT_ROOT, load_dataset
from src.features import split_features_target

RANDOM_STATE = 42
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"


def build_pipeline(features: pd.DataFrame, estimator) -> Pipeline:
    """Fit imputers, scaling and encoding only within each training fold."""
    numeric = features.select_dtypes(include=["number"]).columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ])
    return Pipeline([
        ("preprocessor", preprocessing),
        ("classifier", estimator),
    ])


def main(data_path: str | Path = DEFAULT_DATA_PATH) -> None:
    frame = load_dataset(data_path)
    X, y = split_features_target(frame)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    comparison, fitted = {}, {}
    for name, estimator in candidates.items():
        pipeline = build_pipeline(X_train, estimator)
        scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        comparison[name] = {
            "cv_roc_auc_mean": float(scores.mean()),
            "cv_roc_auc_std": float(scores.std()),
        }
        fitted[name] = pipeline

    selected_name = max(comparison, key=lambda key: comparison[key]["cv_roc_auc_mean"])
    selected_model = fitted[selected_name].fit(X_train, y_train)
    probabilities = selected_model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)
    metrics = {
        "selected_model": selected_name,
        "threshold": 0.50,
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_precision": float(precision_score(y_test, predictions, zero_division=0)),
        "test_recall": float(recall_score(y_test, predictions, zero_division=0)),
        "test_f1": float(f1_score(y_test, predictions, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
        "model_comparison": comparison,
        "test_rows": int(len(y_test)),
        "random_state": RANDOM_STATE,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_model, MODEL_DIR / "customer_churn_pipeline.joblib")
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, display_labels=["Retained", "Churned"], ax=axes[0]
    )
    RocCurveDisplay.from_predictions(y_test, probabilities, ax=axes[1])
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "evaluation.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame({
        "actual_churn": y_test.to_numpy(),
        "churn_probability": probabilities,
    }).to_csv(REPORT_DIR / "test_predictions.csv", index=False)
    print(json.dumps(
        {key: value for key, value in metrics.items() if key != "classification_report"},
        indent=2
    ))


if __name__ == "__main__":
    main()
