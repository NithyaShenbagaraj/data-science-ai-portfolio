"""Create descriptive EDA artifacts for the cached UCI Heart Disease data."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

from src.data import PROJECT_ROOT, feature_types, load_dataset

REPORT_DIR = PROJECT_ROOT / "reports"


def main() -> None:
    """Write cohort, missingness, and feature distribution summaries."""
    X, y = load_dataset()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "rows": int(len(X)),
        "features": int(X.shape[1]),
        "positive_label_rows": int(y.sum()),
        "negative_label_rows": int((y == 0).sum()),
        "positive_label_rate": float(y.mean()),
        "duplicate_rows": int(X.duplicated().sum()),
        "missing_by_feature": {
            str(name): int(count)
            for name, count in X.isna().sum().items()
        },
    }
    (REPORT_DIR / "eda_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    numeric_features, categorical_features = feature_types(X)
    if numeric_features:
        X[numeric_features].describe().transpose().to_csv(
            REPORT_DIR / "numeric_feature_summary.csv"
        )
    if categorical_features:
        category_rows = []
        for column in categorical_features:
            counts = X[column].value_counts(dropna=False)
            category_rows.extend(
                {
                    "feature": column,
                    "value": "Missing" if pd.isna(value) else value,
                    "count": int(count),
                }
                for value, count in counts.items()
            )
        pd.DataFrame(category_rows).to_csv(
            REPORT_DIR / "categorical_feature_counts.csv", index=False
        )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    y.value_counts().sort_index().plot.bar(ax=axes[0], color=["#5B8FF9", "#E8684A"])
    axes[0].set(title="Dataset label counts", xlabel="Label", ylabel="Rows")
    axes[0].set_xticklabels(["Absence (0)", "Presence (1)"], rotation=0)

    missing = X.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        axes[1].text(0.5, 0.5, "No missing feature values", ha="center", va="center")
        axes[1].set_axis_off()
    else:
        missing.plot.bar(ax=axes[1], color="#9270CA")
        axes[1].set(title="Missing values by feature", xlabel="Feature", ylabel="Missing rows")
        axes[1].tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "eda_overview.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"EDA artifacts saved to {REPORT_DIR}")


if __name__ == "__main__":
    main()
