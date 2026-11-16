"""Chronological model selection and held-out evaluation."""
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.data import ROOT
from src.features import CATEGORICAL_FEATURES, FEATURES_FILE, HORIZON_DAYS, NUMERIC_FEATURES, TARGET

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"


def make_pipeline(estimator):
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocess", prep), ("model", estimator)])


def chronological_split(data, embargo_days=HORIZON_DAYS):
    """Split by snapshot date; each later period starts after prior labels mature."""
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(data["snapshot_date"]).dropna().unique()))
    if len(dates) < 8:
        raise ValueError(
            f"Need at least 8 distinct snapshot months for a temporal split; found {len(dates)}."
        )

    train_end_index = max(1, int(len(dates) * 0.45) - 1)
    train_end = dates[train_end_index]
    validation_candidates = dates[dates >= train_end + pd.Timedelta(days=embargo_days)]
    if len(validation_candidates) < 3:
        raise ValueError("Not enough dates remain after the training embargo for validation and test.")

    validation_start = validation_candidates[0]
    validation_start_index = dates.get_loc(validation_start)
    validation_end_index = min(
        validation_start_index + max(1, int(len(dates) * 0.10)),
        len(dates) - 2,
    )
    validation_end = dates[validation_end_index]
    test_candidates = dates[dates >= validation_end + pd.Timedelta(days=embargo_days)]
    if len(test_candidates) == 0:
        raise ValueError("No test dates remain after the validation-label embargo.")

    train = data[pd.to_datetime(data["snapshot_date"]) <= train_end].copy()
    validation = data[pd.to_datetime(data["snapshot_date"]).between(validation_start, validation_end)].copy()
    test = data[pd.to_datetime(data["snapshot_date"]) >= test_candidates[0]].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Temporal split produced an empty partition.")
    return train, validation, test


def evaluate(y, probability):
    predicted = (probability >= 0.5).astype(int)
    result = {
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, predicted).tolist(),
        "positive_rate": float(pd.Series(y).mean()),
    }
    if pd.Series(y).nunique() == 2:
        result["roc_auc"] = float(roc_auc_score(y, probability))
        result["pr_auc"] = float(average_precision_score(y, probability))
    else:
        result.update({"roc_auc": None, "pr_auc": None})
    return result


def main():
    if not FEATURES_FILE.exists():
        raise FileNotFoundError("Run python -m src.features first.")
    data = pd.read_parquet(FEATURES_FILE)
    data["snapshot_date"] = pd.to_datetime(data["snapshot_date"])
    train, validation, test = chronological_split(data)
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=350, min_samples_leaf=4, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=42,
        ),
    }
    validation_metrics = {}
    for name, estimator in candidates.items():
        pipe = make_pipeline(estimator)
        pipe.fit(train[cols], train[TARGET])
        validation_metrics[name] = evaluate(validation[TARGET], pipe.predict_proba(validation[cols])[:, 1])
    selected = max(
        validation_metrics,
        key=lambda key: validation_metrics[key]["pr_auc"] if validation_metrics[key]["pr_auc"] is not None else -1,
    )
    development = pd.concat([train, validation], ignore_index=True)
    final_model = make_pipeline(candidates[selected])
    final_model.fit(development[cols], development[TARGET])
    probabilities = final_model.predict_proba(test[cols])[:, 1]
    test_metrics = evaluate(test[TARGET], probabilities)
    ranked = pd.DataFrame({"actual": test[TARGET].to_numpy(), "score": probabilities})
    capacity = {}
    for fraction in (0.10, 0.20):
        k = max(1, int(len(ranked) * fraction))
        top = ranked.nlargest(k, "score")
        base_rate = ranked.actual.mean()
        capacity[f"top_{int(fraction * 100)}pct"] = {
            "customers_selected": k,
            "precision_at_k": float(top.actual.mean()),
            "lift_at_k": float(top.actual.mean() / base_rate) if base_rate > 0 else None,
        }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_DIR / "repeat_purchase_pipeline.joblib")
    (MODEL_DIR / "feature_columns.json").write_text(json.dumps(cols, indent=2))
    report = {
        "target": "repeat_purchase_90d",
        "definition": "At least one valid purchase in the 90 days after snapshot date",
        "selected_model": selected,
        "train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test),
        "train_period": [str(train.snapshot_date.min().date()), str(train.snapshot_date.max().date())],
        "validation_period": [str(validation.snapshot_date.min().date()), str(validation.snapshot_date.max().date())],
        "test_period": [str(test.snapshot_date.min().date()), str(test.snapshot_date.max().date())],
        "label_horizon_embargo_days": HORIZON_DAYS,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "capacity_metrics": capacity,
        "note": "Historical observational model; not a direct churn label or causal intervention estimate.",
    }
    (REPORT_DIR / "model_metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
