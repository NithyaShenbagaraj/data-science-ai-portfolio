"""Dataset loading and basic validation."""
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"


def load_dataset(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the Telco churn CSV and normalize column names."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download the IBM Telco Customer "
            "Churn CSV and place it in data/raw/."
        )
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()
    required = {TARGET_COLUMN, ID_COLUMN, "tenure", "MonthlyCharges", "TotalCharges"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return frame
