"""Load the project's synthetic Parquet datasets without modifying source files."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]


def resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    if data_dir is not None:
        path = Path(data_dir).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Data directory does not exist: {path}")
        return path
    candidates = [
        PROJECT_DIR / "data" / "processed_candidate",
        PROJECT_DIR / "data" / "processed",
    ]
    for path in candidates:
        if path.exists() and (path / "agency_monthly.parquet").exists():
            return path
    raise FileNotFoundError(
        "No generated Parquet data found. Run src/generate_synthetic_data.py "
        "with --output-dir data/processed_candidate first."
    )


def load_dataset(name: str, data_dir: str | Path | None = None) -> pd.DataFrame:
    allowed = {
        "clients", "contracts", "invoices", "payments",
        "campaigns", "agency_monthly", "campaign_daily",
    }
    if name not in allowed:
        raise ValueError(f"Unknown dataset {name!r}. Expected one of {sorted(allowed)}")
    path = resolve_data_dir(data_dir) / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Required dataset not found: {path}")
    frame = pd.read_parquet(path)
    if frame.empty:
        raise ValueError(f"Dataset is empty: {path}")
    return frame


def load_forecasting_targets(data_dir: str | Path | None = None):
    agency = load_dataset("agency_monthly", data_dir).copy()
    campaign = load_dataset("campaign_daily", data_dir).copy()
    agency["month"] = pd.to_datetime(agency["month"])
    campaign["date"] = pd.to_datetime(campaign["date"])
    agency = agency.sort_values("month").reset_index(drop=True)
    campaign_daily = (
        campaign.groupby("date", as_index=False)
        .agg(
            attributed_campaign_revenue=("attributed_campaign_revenue", "sum"),
            spend_inr=("spend_inr", "sum"),
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
            leads=("leads", "sum"),
            conversions=("conversions", "sum"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    return agency, campaign_daily
