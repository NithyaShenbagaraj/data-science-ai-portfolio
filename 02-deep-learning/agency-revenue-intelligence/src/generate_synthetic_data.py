"""Generate reproducible, clearly labelled synthetic agency and campaign data.

The generator creates separate agency-finance and client-campaign datasets.
It is intended for portfolio engineering and forecasting experiments, not as
a representation of real agency financial performance.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_END_DATE = "2026-10-31"
CHANNELS = ("search", "social", "display", "video", "email", "affiliate")
INDUSTRIES = (
    "ecommerce", "healthcare", "education", "finance", "travel",
    "technology", "real_estate",
)


def _month_end(period: pd.Period) -> pd.Timestamp:
    return period.to_timestamp(how="end").normalize()


def generate_datasets(
    output_dir: str | Path,
    *,
    seed: int = 42,
    n_clients: int = 50,
    campaigns_per_client: int = 2,
    end_date: str = DEFAULT_END_DATE,
) -> dict[str, pd.DataFrame]:
    """Create seven synthetic datasets and write them as Parquet files."""
    if n_clients < 1 or campaigns_per_client < 1:
        raise ValueError("n_clients and campaigns_per_client must be positive")

    rng = np.random.default_rng(seed)
    end = pd.Timestamp(end_date).normalize()
    month_periods = pd.period_range(end=end.to_period("M"), periods=36, freq="M")
    first_month = month_periods[0]
    daily_dates = pd.date_range(end=end, periods=730, freq="D")
    start_day = daily_dates[0]

    # Client master data.
    client_rows = []
    for i in range(n_clients):
        acquired = first_month.to_timestamp() + pd.Timedelta(
            days=int(rng.integers(0, 180))
        )
        client_rows.append({
            "client_id": f"CL{i + 1:04d}",
            "industry": str(rng.choice(INDUSTRIES)),
            "acquisition_date": acquired,
            "monthly_retainer_inr": round(float(rng.uniform(45000, 250000)), 2),
            "synthetic_data": True,
        })
    clients = pd.DataFrame(client_rows)

    # Contracts: annual terms with renewal rows. Some clients eventually churn.
    contract_rows = []
    client_churn_month: dict[str, pd.Period | None] = {}
    client_base_retainer: dict[str, float] = {}
    for row in clients.itertuples(index=False):
        churn_period = None
        if rng.random() < 0.22:
            possible = month_periods[int(rng.integers(18, 35)):]
            if len(possible):
                churn_period = possible[0]
        client_churn_month[row.client_id] = churn_period
        base = float(row.monthly_retainer_inr)
        client_base_retainer[row.client_id] = base
        contract_start = pd.Timestamp(row.acquisition_date).to_period("M")
        contract_number = 1
        while contract_start <= month_periods[-1]:
            if churn_period is not None and contract_start >= churn_period:
                break
            contract_end = min(contract_start + 11, month_periods[-1])
            if churn_period is not None:
                contract_end = min(contract_end, churn_period - 1)
            contract_rows.append({
                "contract_id": f"CT-{row.client_id}-{contract_number:02d}",
                "client_id": row.client_id,
                "contract_type": "monthly_retainer",
                "start_month": str(contract_start),
                "end_month": str(contract_end),
                "monthly_fee_inr": round(base * (1.0 + 0.04 * (contract_number - 1)), 2),
                "renewal_number": contract_number - 1,
                "synthetic_data": True,
            })
            contract_start += 12
            contract_number += 1
    contracts = pd.DataFrame(contract_rows)

    # Monthly earned agency revenue, retained at client-month grain first.
    client_month_rows = []
    for client in clients.itertuples(index=False):
        acquired_period = pd.Timestamp(client.acquisition_date).to_period("M")
        churn_period = client_churn_month[client.client_id]
        for month_index, period in enumerate(month_periods):
            if period < acquired_period or (churn_period is not None and period >= churn_period):
                continue
            contract_number = (period.ordinal - acquired_period.ordinal) // 12 + 1
            # Contracted retainers must reconcile exactly to contract fees.
            retainer = client_base_retainer[client.client_id] * (
                1.0 + 0.04 * (contract_number - 1)
            )
            # Seasonality and noise affect variable project fees, not contracted retainers.
            seasonal = 1.0 + 0.10 * np.sin(2 * np.pi * (period.month - 1) / 12)
            base_project_fee = float(rng.choice([0, 0, 0, 15000, 30000, 50000]))
            project_fee = max(
                0.0,
                base_project_fee * seasonal * float(rng.normal(1.0, 0.15)),
            )
            recognized = retainer + project_fee
            client_month_rows.append({
                "month": _month_end(period),
                "client_id": client.client_id,
                "retainer_revenue_inr": round(retainer, 2),
                "project_fee_revenue_inr": round(project_fee, 2),
                "recognized_agency_revenue": round(recognized, 2),
                "synthetic_data": True,
            })
    client_month = pd.DataFrame(client_month_rows)

    agency_monthly = (
        client_month.groupby("month", as_index=False)
        .agg(
            recognized_agency_revenue=("recognized_agency_revenue", "sum"),
            retainer_revenue_inr=("retainer_revenue_inr", "sum"),
            project_fee_revenue_inr=("project_fee_revenue_inr", "sum"),
            active_clients=("client_id", "nunique"),
        )
    )
    # Derive acquisition/churn indicators without using future observations.
    acquisition_months = clients.assign(
        month=clients["acquisition_date"].dt.to_period("M").dt.to_timestamp(how="end").dt.normalize()
    ).groupby("month").size()
    agency_monthly["new_clients"] = agency_monthly["month"].map(acquisition_months).fillna(0).astype(int)
    agency_monthly["synthetic_data"] = True

    # Invoice records mirror earned fees; payments are separate cash events.
    invoice_rows = []
    payment_rows = []
    for row in client_month.itertuples(index=False):
        invoice_id = f"INV-{row.client_id}-{row.month:%Y%m}"
        amount = float(row.recognized_agency_revenue)
        issue_date = row.month
        due_date = issue_date + pd.Timedelta(days=30)
        invoice_rows.append({
            "invoice_id": invoice_id,
            "client_id": row.client_id,
            "issue_date": issue_date,
            "due_date": due_date,
            "invoice_amount_inr": round(amount, 2),
            "synthetic_data": True,
        })
        if rng.random() < 0.94:
            delay = int(np.clip(rng.gamma(shape=2.0, scale=12.0), 1, 90))
            # Model both full and partial collections; 6% of invoices remain unpaid.
            paid_amount = (
                amount
                if rng.random() < 0.65
                else amount * float(rng.uniform(0.50, 0.95))
            )
            payment_rows.append({
                "payment_id": f"PAY-{invoice_id}",
                "invoice_id": invoice_id,
                "payment_date": issue_date + pd.Timedelta(days=delay),
                "payment_amount_inr": round(paid_amount, 2),
                "synthetic_data": True,
            })
    invoices = pd.DataFrame(invoice_rows)
    payments = pd.DataFrame(payment_rows)

    # Campaign master and daily performance, at campaign-day grain.
    campaign_rows = []
    daily_rows = []
    campaign_count = n_clients * campaigns_per_client
    for campaign_index in range(campaign_count):
        client_id = f"CL{int(rng.integers(1, n_clients + 1)):04d}"
        channel = str(rng.choice(CHANNELS))
        campaign_start = start_day + pd.Timedelta(
            days=int(rng.integers(0, max(1, len(daily_dates) - 120)))
        )
        duration = int(rng.integers(90, 366))
        campaign_end = min(campaign_start + pd.Timedelta(days=duration - 1), end)
        daily_budget = float(rng.uniform(800, 12000))
        campaign_id = f"CAM{campaign_index + 1:04d}"
        campaign_rows.append({
            "campaign_id": campaign_id,
            "client_id": client_id,
            "channel": channel,
            "campaign_start": campaign_start,
            "campaign_end": campaign_end,
            "daily_budget_inr": round(daily_budget, 2),
            "synthetic_data": True,
        })

        channel_ctr = {
            "search": 0.035, "social": 0.018, "display": 0.009,
            "video": 0.012, "email": 0.045, "affiliate": 0.025,
        }[channel]
        channel_conversion = {
            "search": 0.045, "social": 0.025, "display": 0.012,
            "video": 0.018, "email": 0.055, "affiliate": 0.035,
        }[channel]
        active_dates = daily_dates[
            (daily_dates >= campaign_start) & (daily_dates <= campaign_end)
        ]
        campaign_scale = float(rng.lognormal(mean=0.0, sigma=0.18))
        for day in active_dates:
            weekday_factor = 0.82 if day.dayofweek >= 5 else 1.0
            seasonal_factor = 1.0 + 0.12 * np.sin(2 * np.pi * (day.dayofyear - 1) / 365.25)
            spend = max(0.0, daily_budget * weekday_factor * float(rng.normal(1, 0.12)))
            cpc = float(rng.uniform(8, 45))
            clicks = int(max(0, rng.poisson(spend / cpc)))
            ctr = max(0.001, channel_ctr * float(rng.lognormal(0, 0.12)))
            impressions = int(max(clicks, clicks / ctr)) if clicks else 0
            leads = int(rng.binomial(clicks, min(channel_conversion * 1.8, 0.5)))
            conversions = int(rng.binomial(leads, min(channel_conversion * 0.65, 0.8))) if leads else 0
            average_order_value = float(rng.uniform(900, 6500))
            attributed_revenue = (
                conversions * average_order_value * campaign_scale * seasonal_factor
            )
            daily_rows.append({
                "date": day,
                "campaign_id": campaign_id,
                "client_id": client_id,
                "channel": channel,
                "spend_inr": round(spend, 2),
                "impressions": impressions,
                "clicks": clicks,
                "leads": leads,
                "conversions": conversions,
                "attributed_campaign_revenue": round(attributed_revenue, 2),
                "synthetic_data": True,
            })

    campaigns = pd.DataFrame(campaign_rows)
    campaign_daily = pd.DataFrame(daily_rows)

    datasets = {
        "clients": clients,
        "contracts": contracts,
        "invoices": invoices,
        "payments": payments,
        "campaigns": campaigns,
        "agency_monthly": agency_monthly,
        "campaign_daily": campaign_daily,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in datasets.items():
        frame.to_parquet(output / f"{name}.parquet", index=False)
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Output directory for generated Parquet datasets.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clients", type=int, default=50)
    parser.add_argument("--campaigns-per-client", type=int, default=2)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    args = parser.parse_args()
    datasets = generate_datasets(
        args.output_dir,
        seed=args.seed,
        n_clients=args.clients,
        campaigns_per_client=args.campaigns_per_client,
        end_date=args.end_date,
    )
    for name, frame in datasets.items():
        print(f"{name}: {len(frame):,} rows")


if __name__ == "__main__":
    main()
