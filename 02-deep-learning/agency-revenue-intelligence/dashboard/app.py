"""Streamlit dashboard for validated actuals, forecasts and channel diagnostics."""
from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[2]
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

metrics_path = PROJECT_DIR / "reports" / "generated" / "test_metrics.csv"
predictions_path = PROJECT_DIR / "reports" / "generated" / "test_predictions.csv"
c1, c2, c3 = st.columns(3)
c1.metric("Latest agency revenue", f"₹{agency.recognized_agency_revenue.iloc[-1]:,.0f}")
c2.metric("Active clients", f"{int(agency.active_clients.iloc[-1]):,}" if "active_clients" in agency else "—")
c3.metric("Campaign attributed revenue (daily avg)", f"₹{campaign.attributed_campaign_revenue.mean():,.0f}")

tab1, tab2, tab3 = st.tabs(["Agency revenue", "Campaign performance", "Forecast evaluation"])
with tab1:
    fig = px.line(agency, x="month", y=["recognized_agency_revenue", "retainer_revenue_inr", "project_fee_revenue_inr"], title="Monthly agency revenue composition")
    st.plotly_chart(fig, use_container_width=True)
    fig_clients = px.line(agency, x="month", y=["active_clients", "new_clients"], title="Client activity")
    st.plotly_chart(fig_clients, use_container_width=True)

with tab2:
    daily_fig = px.line(campaign, x="date", y=["attributed_campaign_revenue", "spend_inr"], title="Daily campaign revenue and spend")
    st.plotly_chart(daily_fig, use_container_width=True)
    campaign_daily = load_dataset("campaign_daily")
    channel = campaign_daily.groupby("channel", as_index=False).agg(
        spend=("spend_inr", "sum"),
        revenue=("attributed_campaign_revenue", "sum"),
        clicks=("clicks", "sum"), leads=("leads", "sum"),
        conversions=("conversions", "sum"),
    )
    channel["roas"] = channel["revenue"] / channel["spend"].replace(0, float("nan"))
    channel["click_to_lead_rate"] = channel["leads"] / channel["clicks"].replace(0, float("nan"))
    channel["lead_to_customer_rate"] = channel["conversions"] / channel["leads"].replace(0, float("nan"))
    st.subheader("Channel performance")
    st.dataframe(channel.sort_values("roas", ascending=False), use_container_width=True)
    st.caption("ROAS is attributed revenue divided by advertising spend; it is not profit or causal return.")

with tab3:
    if metrics_path.exists() and predictions_path.exists():
        st.subheader("Held-out test metrics")
        st.dataframe(pd.read_csv(metrics_path), use_container_width=True)
        predictions = pd.read_csv(predictions_path, parse_dates=["date"])
        fig = px.line(predictions, x="date", y=["actual", "predicted"], color="target", facet_row="target", title="Actual vs predicted — test period")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Forecast artifacts are not available yet. Run python -m src.train_forecasts first.")

st.divider()
st.subheader("Illustrative campaign budget scenario")
st.caption("This is a transparent arithmetic scenario, not a causal forecast.")
budget_change = st.slider("Assumed budget change (%)", min_value=-50, max_value=100, value=10, step=5)
historical_roas = campaign.attributed_campaign_revenue.sum() / campaign.spend_inr.sum()
base_spend = float(campaign.spend_inr.sum())
scenario_spend = base_spend * (1 + budget_change / 100)
scenario_revenue = scenario_spend * historical_roas
st.metric("Illustrative attributed revenue", f"₹{scenario_revenue:,.0f}", delta=f"{budget_change:+d}% budget assumption")
st.write(f"Uses a fixed historical blended ROAS of {historical_roas:.2f}x. It assumes no saturation, mix shift, lag, or diminishing returns.")
