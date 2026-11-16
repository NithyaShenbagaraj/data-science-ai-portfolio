
"""Tune healthcare classification models without modifying the baseline."""
from __future__ import annotations

import json

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from src.data import PROJECT_ROOT, feature_types, load_dataset
from src.train import (
    RANDOM_STATE,
    TEST_SIZE,
    build_preprocessor,
    calculate_metrics,
)

REPORT_DIR = PROJECT_ROOT / "reports"
TUNED_MODEL_PATH = (
    PROJECT_ROOT / "models" / "tuned_healthcare_risk_pipeline.joblib"
)


def build_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    estimator,
) -> Pipeline:
    """Build a pipeline using the project's existing preprocessing."""
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(numeric_features, categorical_features),
            ),
            ("classifier", estimator),
        ]
    )


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    TUNED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load and split data.
    X, y = load_dataset()
    numeric_features, categorical_features = feature_types(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # 2. Define cross-validation.
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # 3. Logistic Regression: exhaustive grid search.
    logistic_pipeline = build_pipeline(
        numeric_features,
        categorical_features,
        LogisticRegression(
            solver="liblinear",
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
    )

    logistic_grid = {
        "classifier__C": [0.01, 0.1, 1, 10, 100],
        "classifier__class_weight": [None, "balanced"],
    }

    logistic_search = GridSearchCV(
        estimator=logistic_pipeline,
        param_grid=logistic_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
        refit=True,
    )

    print("Tuning Logistic Regression...")
    logistic_search.fit(X_train, y_train)

    # 4. Random Forest: randomized search.
    forest_pipeline = build_pipeline(
        numeric_features,
        categorical_features,
        RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    )

    forest_distributions = {
        "classifier__n_estimators": [200, 400, 600],
        "classifier__max_depth": [None, 3, 5, 8],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", 0.8],
        "classifier__class_weight": [
            None,
            "balanced",
            "balanced_subsample",
        ],
    }

    forest_search = RandomizedSearchCV(
        estimator=forest_pipeline,
        param_distributions=forest_distributions,
        n_iter=30,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=True,
    )

    print("\nTuning Random Forest...")
    forest_search.fit(X_train, y_train)

    # Save detailed cross-validation results for analysis.
    pd.DataFrame(logistic_search.cv_results_).to_csv(
        REPORT_DIR / "logistic_tuning_cv_results.csv",
        index=False,
    )

    pd.DataFrame(forest_search.cv_results_).to_csv(
        REPORT_DIR / "random_forest_tuning_cv_results.csv",
        index=False,
    )

    # 5. Compare the best cross-validation results.
    searches = {
        "Tuned Logistic Regression": logistic_search,
        "Tuned Random Forest": forest_search,
    }

    comparison_rows = []

    for name, search in searches.items():
        comparison_rows.append(
            {
                "model": name,
                "best_cv_roc_auc": float(search.best_score_),
                "best_parameters": json.dumps(
                    search.best_params_,
                    default=str,
                ),
            }
        )

        print(f"\n{name}")
        print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
        print(f"Best parameters: {search.best_params_}")

    comparison = pd.DataFrame(comparison_rows).sort_values(
        "best_cv_roc_auc",
        ascending=False,
    )

    comparison.to_csv(
        REPORT_DIR / "tuning_comparison.csv",
        index=False,
    )

    # 6. Select the best tuned model using CV results.
    best_name = str(comparison.iloc[0]["model"])
    best_search = searches[best_name]
    best_pipeline = best_search.best_estimator_

    # 7. Evaluate once on the untouched holdout set.
    y_probability = best_pipeline.predict_proba(X_test)[:, 1]
    y_prediction = (y_probability >= 0.5).astype(int)

    metrics = calculate_metrics(
        y_test,
        y_prediction,
        y_probability,
    )

    metrics["selected_model"] = best_name
    metrics["best_cv_roc_auc"] = float(best_search.best_score_)
    metrics["best_parameters"] = best_search.best_params_
    metrics["holdout_rows"] = int(len(y_test))
    metrics["classification_report"] = classification_report(
        y_test,
        y_prediction,
        output_dict=True,
        zero_division=0,
    )

    (REPORT_DIR / "tuned_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str),
        encoding="utf-8",
    )

    # 8. Save the tuned model separately from the baseline.
    bundle = {
        "pipeline": best_pipeline,
        "feature_columns": list(X.columns),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "selected_model": best_name,
        "threshold": 0.5,
        "best_parameters": best_search.best_params_,
    }

    joblib.dump(bundle, TUNED_MODEL_PATH)

    print("\n--- TUNING COMPLETE ---")
    print("Selected model:", best_name)
    print("Holdout metrics:")
    print(
        json.dumps(
            {
                key: value
                for key, value in metrics.items()
                if key != "classification_report"
            },
            indent=2,
            default=str,
        )
    )
    print("\nSaved comparison:", REPORT_DIR / "tuning_comparison.csv")
    print("Saved metrics:", REPORT_DIR / "tuned_metrics.json")
    print("Saved model:", TUNED_MODEL_PATH)


if __name__ == "__main__":
    main()
