"""Train, compare, evaluate, and persist healthcare classification models."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import PROJECT_ROOT, feature_types, load_dataset

RANDOM_STATE = 42
TEST_SIZE = 0.20
MODEL_PATH = PROJECT_ROOT / "models" / "healthcare_risk_pipeline.joblib"
REPORT_DIR = PROJECT_ROOT / "reports"


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def build_models() -> dict[str, object]:
    models: dict[str, object] = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=4,
        )
    except ImportError:
        print("XGBoost is not installed; comparing Logistic Regression and Random Forest.")

    return models


def calculate_metrics(y_true: pd.Series, y_pred, y_probability) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    return {
        "roc_auc": float(roc_auc_score(y_true, y_probability)),
        "average_precision": float(average_precision_score(y_true, y_probability)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": specificity,
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset()
    numeric_features, categorical_features = feature_types(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    comparison_rows = []
    fitted_candidates = {}

    for name, estimator in build_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
                ("classifier", estimator),
            ]
        )
        scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1
        )
        pipeline.fit(X_train, y_train)
        fitted_candidates[name] = pipeline
        comparison_rows.append(
            {
                "model": name,
                "cv_roc_auc_mean": float(scores.mean()),
                "cv_roc_auc_std": float(scores.std()),
            }
        )
        print(f"{name}: CV ROC-AUC {scores.mean():.3f} ± {scores.std():.3f}")

    comparison = pd.DataFrame(comparison_rows).sort_values(
        "cv_roc_auc_mean", ascending=False
    )
    best_name = str(comparison.iloc[0]["model"])
    best_pipeline = fitted_candidates[best_name]

    # The holdout set is used only after model selection.
    y_probability = best_pipeline.predict_proba(X_test)[:, 1]
    y_prediction = (y_probability >= 0.5).astype(int)
    metrics = calculate_metrics(y_test, y_prediction, y_probability)
    metrics["selected_model"] = best_name
    metrics["holdout_rows"] = int(len(y_test))
    metrics["positive_class_rows"] = int(y.sum())
    metrics["total_rows"] = int(len(y))
    metrics["classification_report"] = classification_report(
        y_test, y_prediction, output_dict=True, zero_division=0
    )

    comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    (REPORT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "actual": y_test.to_numpy(),
            "predicted": y_prediction,
            "predicted_probability": y_probability,
        },
        index=y_test.index,
    ).to_csv(REPORT_DIR / "test_predictions.csv", index_label="row_id")

    cm = confusion_matrix(y_test, y_prediction, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    display = __import__("sklearn.metrics", fromlist=["ConfusionMatrixDisplay"]).ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["No disease label", "Disease label"]
    )
    display.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title("Holdout confusion matrix")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, y_probability, ax=ax)
    ax.set_title(f"Holdout ROC curve — {best_name}")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "roc_curve.png", dpi=160)
    plt.close(fig)

    bundle = {
        "pipeline": best_pipeline,
        "feature_columns": list(X.columns),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "selected_model": best_name,
        "threshold": 0.5,
    }
    joblib.dump(bundle, MODEL_PATH)

    print("\nSelected model:", best_name)
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved reports: {REPORT_DIR}")


if __name__ == "__main__":
    main()
