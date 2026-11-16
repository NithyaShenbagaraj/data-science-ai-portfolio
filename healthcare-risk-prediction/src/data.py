"""Data acquisition and validation for the UCI Heart Disease dataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "heart_disease.csv"
TARGET_COLUMN = "target"


def download_dataset(output_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Download UCI dataset 45, normalize its target, and cache a local CSV."""
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'ucimlrepo'. Install requirements.txt first."
        ) from exc

    dataset = fetch_ucirepo(id=45)
    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()

    if targets.shape[1] == 0:
        raise ValueError("The UCI response did not contain a target column.")

    target = pd.to_numeric(targets.iloc[:, 0], errors="coerce")
    frame = features.copy()
    frame[TARGET_COLUMN] = target
    frame = frame.dropna(subset=[TARGET_COLUMN]).copy()
    frame[TARGET_COLUMN] = (frame[TARGET_COLUMN] > 0).astype("int64")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def load_dataset(path: Path = RAW_DATA_PATH, refresh: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """Load cached data or download it, then return features and binary target."""
    if refresh or not path.exists():
        frame = download_dataset(path)
    else:
        frame = pd.read_csv(path)

    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}'.")

    frame = frame.replace("?", pd.NA)
    y = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    valid_target = y.notna()
    X = frame.loc[valid_target].drop(columns=[TARGET_COLUMN]).copy()
    y = (y.loc[valid_target] > 0).astype("int64")

    # UCI encodes several categorical clinical attributes as integer codes.
    # Convert every feature to numeric while preserving missing values.
    for column in X.columns:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    if X.empty:
        raise ValueError("The dataset contains no usable feature rows.")
    if y.nunique() != 2:
        raise ValueError("Expected a binary target with both classes represented.")
    if not X.index.equals(y.index):
        raise ValueError("Feature and target indices are misaligned.")

    return X, y


def feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return numeric and categorical columns using the UCI variable definitions."""
    categorical_candidates = {
        "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
    }
    categorical = [c for c in X.columns if c.lower() in categorical_candidates]
    numeric = [c for c in X.columns if c not in categorical]
    return numeric, categorical
