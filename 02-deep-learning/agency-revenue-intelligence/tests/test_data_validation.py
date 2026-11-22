import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = Path(
    os.environ.get(
        "AGENCY_DATA_DIR",
        PROJECT_DIR / "data" / "processed_candidate",
    )
)

EXPECTED_FILES = {
    "clients": "clients.parquet",
    "contracts": "contracts.parquet",
    "invoices": "invoices.parquet",
    "payments": "payments.parquet",
    "campaigns": "campaigns.parquet",
    "agency_monthly": "agency_monthly.parquet",
    "campaign_daily": "campaign_daily.parquet",
}


@pytest.fixture(scope="module")
def datasets():
    missing = [
        name for name, filename in EXPECTED_FILES.items()
        if not (DATA_DIR / filename).exists()
    ]

    assert not missing, (
        f"Missing datasets in {DATA_DIR}: {missing}"
    )

    return {
        name: pd.read_parquet(DATA_DIR / filename)
        for name, filename in EXPECTED_FILES.items()
    }


def test_all_datasets_have_rows_and_no_missing_values(datasets):
    for name, df in datasets.items():
        assert not df.empty, f"{name} is empty"
        assert not df.isnull().any().any(), (
            f"{name} contains missing values"
        )


def test_all_datasets_are_marked_synthetic(datasets):
    for name, df in datasets.items():
        assert "synthetic_data" in df.columns
        assert df["synthetic_data"].eq(True).all(), (
            f"{name} contains non-synthetic records"
        )


def test_primary_keys_are_unique(datasets):
    primary_keys = {
        "clients": ["client_id"],
        "contracts": ["contract_id"],
        "invoices": ["invoice_id"],
        "payments": ["payment_id"],
        "campaigns": ["campaign_id"],
    }

    for name, columns in primary_keys.items():
        assert not datasets[name].duplicated(columns).any(), (
            f"Duplicate primary keys in {name}"
        )

    assert not datasets["campaign_daily"].duplicated(
        ["campaign_id", "date"]
    ).any(), "Duplicate campaign-day records"


def test_foreign_keys_reference_existing_records(datasets):
    clients = set(datasets["clients"]["client_id"])
    invoices = set(datasets["invoices"]["invoice_id"])
    campaigns = set(datasets["campaigns"]["campaign_id"])

    for name in ["contracts", "invoices", "campaigns"]:
        assert datasets[name]["client_id"].isin(clients).all(), (
            f"Invalid client reference in {name}"
        )

    assert datasets["payments"]["invoice_id"].isin(invoices).all()
    assert datasets["campaign_daily"]["campaign_id"].isin(campaigns).all()
    assert datasets["campaign_daily"]["client_id"].isin(clients).all()


def test_client_master_data_is_valid(datasets):
    clients = datasets["clients"]

    assert clients["client_id"].str.startswith("CL").all()
    assert clients["industry"].notna().all()
    assert clients["monthly_retainer_inr"].gt(0).all()

    assert clients["acquisition_date"].between(
        pd.Timestamp("2023-11-01"),
        pd.Timestamp("2026-10-31"),
    ).all()


def test_contract_dates_renewals_and_fees_are_consistent(datasets):
    clients = datasets["clients"].set_index("client_id")
    contracts = datasets["contracts"].copy()

    assert contracts["contract_type"].eq("monthly_retainer").all()
    assert contracts["monthly_fee_inr"].gt(0).all()
    assert contracts["renewal_number"].ge(0).all()

    contracts["start_period"] = pd.PeriodIndex(
        contracts["start_month"], freq="M"
    )
    contracts["end_period"] = pd.PeriodIndex(
        contracts["end_month"], freq="M"
    )

    assert (contracts["end_period"] >= contracts["start_period"]).all()

    for client_id, group in contracts.groupby("client_id"):
        group = group.sort_values("renewal_number")
        client = clients.loc[client_id]

        renewals = group["renewal_number"].tolist()
        assert renewals == list(range(len(group))), (
            f"Non-sequential renewals for {client_id}"
        )

        acquired_period = client["acquisition_date"].to_period("M")

        for row in group.itertuples():
            expected_start = acquired_period + 12 * row.renewal_number

            assert row.start_period == expected_start
            assert row.end_period <= row.start_period + 11

            expected_fee = round(
                client["monthly_retainer_inr"]
                * (1 + 0.04 * row.renewal_number),
                2,
            )

            assert row.monthly_fee_inr == pytest.approx(
                expected_fee, abs=0.01
            )


def test_invoice_dates_and_amounts_are_valid(datasets):
    invoices = datasets["invoices"]

    assert invoices["invoice_id"].is_unique
    assert invoices["invoice_amount_inr"].gt(0).all()
    assert (invoices["due_date"] >= invoices["issue_date"]).all()

    assert invoices["invoice_id"].str.startswith("INV-").all()



def test_invoice_amounts_reconcile_to_monthly_recognized_revenue(
    datasets,
):
    invoices = datasets["invoices"]
    agency = datasets["agency_monthly"]

    invoice_monthly = (
        invoices.groupby("issue_date")["invoice_amount_inr"]
        .sum()
    )

    agency_monthly = (
        agency.set_index("month")["recognized_agency_revenue"]
        .sort_index()
    )

    # Include months with zero invoices as zero invoiced revenue.
    invoice_monthly = invoice_monthly.reindex(
        agency_monthly.index,
        fill_value=0,
    )

    pd.testing.assert_series_equal(
        invoice_monthly,
        agency_monthly,
        check_names=False,
        check_exact=False,
        rtol=0,
        atol=0.02,
    )



def test_payments_are_valid_and_do_not_exceed_invoice_amounts(
    datasets,
):
    payments = datasets["payments"]
    invoices = datasets["invoices"].set_index("invoice_id")

    assert payments["payment_id"].is_unique
    assert payments["payment_amount_inr"].gt(0).all()

    payment_invoice_dates = payments["invoice_id"].map(
        invoices["issue_date"]
    )

    assert (
        payments["payment_date"] >= payment_invoice_dates
    ).all()

    collected = payments.groupby("invoice_id")[
        "payment_amount_inr"
    ].sum()

    invoice_amounts = invoices.loc[
        collected.index, "invoice_amount_inr"
    ]

    assert (collected <= invoice_amounts + 0.01).all(), (
        "Collected payments exceed invoice amounts"
    )


def test_agency_monthly_has_36_consecutive_months(datasets):
    agency = datasets["agency_monthly"].sort_values("month")

    expected_months = (
        pd.period_range(
            start="2023-11",
            periods=36,
            freq="M",
        )
        .to_timestamp(how="end")
        .normalize()
    )

    assert len(agency) == 36
    assert agency["month"].tolist() == expected_months.tolist()


def test_agency_revenue_components_reconcile(datasets):
    agency = datasets["agency_monthly"]

    calculated = (
        agency["retainer_revenue_inr"]
        + agency["project_fee_revenue_inr"]
    )

    np.testing.assert_allclose(
        agency["recognized_agency_revenue"],
        calculated,
        rtol=0,
        atol=0.02,
    )

    assert agency["retainer_revenue_inr"].ge(0).all()
    assert agency["project_fee_revenue_inr"].ge(0).all()
    assert agency["recognized_agency_revenue"].ge(0).all()
    assert agency["active_clients"].ge(0).all()
    assert agency["new_clients"].ge(0).all()

    assert (
        agency["new_clients"] <= agency["active_clients"]
    ).all()


def test_campaign_master_data_is_valid(datasets):
    campaigns = datasets["campaigns"]
    clients = datasets["clients"].set_index("client_id")

    assert campaigns["campaign_id"].is_unique
    assert campaigns["daily_budget_inr"].gt(0).all()

    assert campaigns["channel"].isin(
        ["search", "social", "display", "video", "email", "affiliate"]
    ).all()

    assert (
        campaigns["campaign_end"] >= campaigns["campaign_start"]
    ).all()

    acquisition_dates = campaigns["client_id"].map(
        clients["acquisition_date"]
    )

    assert (
        campaigns["campaign_start"] >= acquisition_dates
    ).all()


def test_campaign_daily_records_match_campaign_master(datasets):
    daily = datasets["campaign_daily"]
    campaigns = datasets["campaigns"].set_index("campaign_id")

    start_dates = daily["campaign_id"].map(
        campaigns["campaign_start"]
    )
    end_dates = daily["campaign_id"].map(
        campaigns["campaign_end"]
    )

    assert (daily["date"] >= start_dates).all()
    assert (daily["date"] <= end_dates).all()

    assert daily["client_id"].eq(
        daily["campaign_id"].map(campaigns["client_id"])
    ).all()

    assert daily["channel"].eq(
        daily["campaign_id"].map(campaigns["channel"])
    ).all()


def test_campaign_funnel_metrics_are_consistent(datasets):
    daily = datasets["campaign_daily"]

    funnel_columns = [
        "impressions",
        "clicks",
        "leads",
        "conversions",
    ]

    for column in funnel_columns:
        assert daily[column].ge(0).all(), (
            f"Negative values in {column}"
        )

    assert (daily["clicks"] <= daily["impressions"]).all()
    assert (daily["leads"] <= daily["clicks"]).all()
    assert (daily["conversions"] <= daily["leads"]).all()


def test_campaign_spend_and_attributed_revenue_are_valid(datasets):
    daily = datasets["campaign_daily"]

    assert daily["spend_inr"].ge(0).all()
    assert daily["attributed_campaign_revenue"].ge(0).all()

    zero_conversion = daily["conversions"].eq(0)

    assert daily.loc[
        zero_conversion, "attributed_campaign_revenue"
    ].eq(0).all()


def test_each_campaign_has_daily_performance_records(datasets):
    campaigns = set(datasets["campaigns"]["campaign_id"])
    daily_campaigns = set(
        datasets["campaign_daily"]["campaign_id"]
    )

    assert daily_campaigns == campaigns
