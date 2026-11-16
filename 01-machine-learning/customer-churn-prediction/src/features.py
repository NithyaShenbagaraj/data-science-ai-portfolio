"""Feature engineering and target separation."""
import pandas as pd
from src.data import ID_COLUMN, TARGET_COLUMN

SERVICE_COLUMNS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create model-ready customer features without using target or ID."""
    data = frame.copy()
    data.columns = data.columns.str.strip()
    if "TotalCharges" in data:
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    if "tenure" in data:
        data["tenure"] = pd.to_numeric(data["tenure"], errors="coerce")
        data["tenure_group"] = pd.cut(
            data["tenure"], bins=[-1, 12, 24, 48, 72],
            labels=["0-12", "13-24", "25-48", "49-72"]
        ).astype("object")
    available = [column for column in SERVICE_COLUMNS if column in data.columns]
    if available:
        data["service_count"] = data[available].apply(
            lambda row: sum(value == "Yes" for value in row), axis=1
        )
    return data.drop(columns=[ID_COLUMN, TARGET_COLUMN], errors="ignore")


def split_features_target(frame: pd.DataFrame):
    """Return engineered features and a binary churn target."""
    if TARGET_COLUMN not in frame:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is required.")
    target = frame[TARGET_COLUMN].astype(str).str.strip().map({"No": 0, "Yes": 1})
    if target.isna().any():
        invalid = sorted(frame.loc[target.isna(), TARGET_COLUMN].astype(str).unique())
        raise ValueError(f"Unexpected target labels: {invalid}")
    return engineer_features(frame), target.astype(int)
