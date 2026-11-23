"""Streamlit dashboard for revenue forecasting and model explainability."""
from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.data_access import load_dataset, load_forecasting_targets

st.set_page_config(page_title="Agency Revenue Intelligence", layout="wide")
st.title("Agency Revenue Intelligence & Forecasting")
st.caption("Synthetic portfolio data. Forecasts and scenarios are estimates, not guarantees.")


@st.cache_data
def load_data():
    agency, campaign = load_forecasting_targets()
    campaigns = load_dataset("campaigns")
    return agency, campaign, campaigns


try:
    agency, campaign, campaigns = load_data()
except Exception as exc:
    st.error(f"Unable to load project data: {exc}")
    st.info("Generate the synthetic datasets into data/processed_candidate before launching the dashboard.")
    st.stop()

FORECAST_DIR = PROJECT_DIR / "reports" / "forecast_results"
EXPLAIN_DIR = PROJECT_DIR / "reports" / "explainability"

metrics_path = FORECAST_DIR / "test_metrics.csv"
validation_path = FORECAST_DIR / "validation_metrics.csv"
predictions_path = FORECAST_DIR / "test_predictions.csv"

c1, c2, c3 = st.columns(3)
c1.metric("Latest agency revenue", f"₹{agency.recognized_agency_revenue.iloc[-1]:,.0f}")
c2.metric("Active clients", f"{int(agency.active_clients.iloc[-1]):,}" if "active_clients" in agency else "—")
c3.metric("Campaign attributed revenue (daily avg)", f"₹{campaign.attributed_campaign_revenue.mean():,.0f}")

tab1, tab2, tab3, tab4 = st.tabs([
    "Agency revenue",
    "Campaign performance",
    "Forecast evaluation",
    "Model explainability",
])

with tab1:
    fig = px.line(
        agency,
        x="month",
        y=["recognized_agency_revenue", "retainer_revenue_inr", "project_fee_revenue_inr"],
        title="Monthly agency revenue composition",
    )
    st.plotly_chart(fig, width="stretch")

    fig_clients = px.line(
        agency,
        x="month",
        y=["active_clients", "new_clients"],
        title="Client activity",
    )
    st.plotly_chart(fig_clients, width="stretch")

with tab2:
    daily_fig = px.line(
        campaign,
        x="date",
        y=["attributed_campaign_revenue", "spend_inr"],
        title="Daily campaign revenue and spend",
    )
    st.plotly_chart(daily_fig, width="stretch")

    campaign_daily = load_dataset("campaign_daily")
    channel = campaign_daily.groupby("channel", as_index=False).agg(
        spend=("spend_inr", "sum"),
        revenue=("attributed_campaign_revenue", "sum"),
        clicks=("clicks", "sum"),
        leads=("leads", "sum"),
        conversions=("conversions", "sum"),
    )
    channel["roas"] = channel["revenue"] / channel["spend"].replace(0, float("nan"))
    channel["click_to_lead_rate"] = channel["leads"] / channel["clicks"].replace(0, float("nan"))
    channel["lead_to_customer_rate"] = channel["conversions"] / channel["leads"].replace(0, float("nan"))

    st.subheader("Channel performance")
    st.dataframe(channel.sort_values("roas", ascending=False), width="stretch")
    st.caption("ROAS is attributed revenue divided by advertising spend; it is not profit or causal return.")

with tab3:
    if metrics_path.exists() and predictions_path.exists():
        st.subheader("Held-out test metrics")
        test_metrics = pd.read_csv(metrics_path)
        st.dataframe(test_metrics, width="stretch")

        if validation_path.exists():
            st.subheader("Validation model comparison")
            validation_metrics = pd.read_csv(validation_path)
            st.dataframe(
                validation_metrics.sort_values(["target", "wape"]),
                width="stretch",
            )

        predictions = pd.read_csv(predictions_path, parse_dates=["date"])
        fig = px.line(
            predictions,
            x="date",
            y=["actual", "predicted"],
            color="target",
            facet_row="target",
            title="Actual vs predicted — held-out test period",
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Models are selected using validation WAPE. Test metrics are reported "
            "separately on the held-out chronological period."
        )
    else:
        st.warning("Forecast artifacts are not available yet. Run python -m src.train_forecasts first.")

with tab4:
    importance_path = EXPLAIN_DIR / "shap_feature_importance.csv"
    observations_path = EXPLAIN_DIR / "shap_observation_values.csv"
    summary_path = EXPLAIN_DIR / "shap_summary.png"

    if importance_path.exists() and observations_path.exists() and summary_path.exists():
        importance = pd.read_csv(importance_path)
        observations = pd.read_csv(observations_path, parse_dates=["date"])

        st.subheader("Global feature importance")
        st.write(
            "Mean absolute SHAP measures the average magnitude of each feature's "
            "contribution to the model output. Larger values indicate greater "
            "influence on this fitted model's predictions."
        )

        importance_fig = px.bar(
            importance.sort_values("mean_abs_shap"),
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            title="Mean absolute SHAP by feature",
            labels={
                "mean_abs_shap": "Mean absolute SHAP value",
                "feature": "Feature",
            },
        )
        st.plotly_chart(importance_fig, width="stretch")

        st.subheader("SHAP summary — contribution distribution")
        st.image(str(summary_path), width="stretch")

        st.divider()
        st.subheader("Individual historical observation")

        date_options = observations["date"].dt.strftime("%Y-%m-%d").tolist()
        selected_date = st.selectbox(
            "Select a historical date",
            options=date_options,
            index=len(date_options) - 1,
        )

        selected = observations.loc[
            observations["date"].dt.strftime("%Y-%m-%d") == selected_date
        ].iloc[0]

        st.metric(
            "Actual attributed campaign revenue",
            f"₹{selected['attributed_campaign_revenue']:,.0f}",
        )

        shap_columns = [
            column for column in observations.columns
            if column.startswith("shap_")
        ]
        local = pd.DataFrame({
            "feature": [column.replace("shap_", "", 1) for column in shap_columns],
            "shap_value": [selected[column] for column in shap_columns],
        })
        local["direction"] = local["shap_value"].apply(
            lambda value: "Positive contribution" if value >= 0 else "Negative contribution"
        )
        local = local.sort_values(
            "shap_value",
            key=lambda values: values.abs(),
        )

        local_fig = px.bar(
            local,
            x="shap_value",
            y="feature",
            orientation="h",
            color="direction",
            title=f"Feature contributions for {selected_date}",
            labels={
                "shap_value": "SHAP contribution",
                "feature": "Feature",
                "direction": "Contribution direction",
            },
        )
        st.plotly_chart(local_fig, width="stretch")

        st.caption(
            "Positive SHAP values increase the model output relative to its baseline; "
            "negative values decrease it. These are model contributions, not causal effects."
        )
        st.info(
            "Scope: explanations use historical one-step-ahead feature rows from "
            "2024-12-04 through 2026-07-13. The held-out test period was excluded "
            "from model fitting and SHAP calculation. These explanations do not "
            "describe the recursive test forecast."
        )
    else:
        st.warning(
            "SHAP artifacts are missing. Run "
            "`python -m src.generate_explanations` to generate them."
        )

st.divider()
st.subheader("Illustrative campaign budget scenario")
st.caption("This is a transparent arithmetic scenario, not a causal forecast.")

budget_change = st.slider(
    "Assumed budget change (%)",
    min_value=-50,
    max_value=100,
    value=10,
    step=5,
)
historical_roas = campaign.attributed_campaign_revenue.sum() / campaign.spend_inr.sum()
base_spend = float(campaign.spend_inr.sum())
scenario_spend = base_spend * (1 + budget_change / 100)
scenario_revenue = scenario_spend * historical_roas

st.metric(
    "Illustrative attributed revenue",
    f"₹{scenario_revenue:,.0f}",
    delta=f"{budget_change:+d}% budget assumption",
)
st.write(
    f"Uses a fixed historical blended ROAS of {historical_roas:.2f}x. "
    "It assumes no saturation, mix shift, lag, or diminishing returns."
)
