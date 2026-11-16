"""Streamlit app for repeat-purchase propensity scoring."""
import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "repeat_purchase_pipeline.joblib"
METRICS_PATH = ROOT / "reports" / "model_metrics.json"
NUMERIC = [
    "recency_days", "invoice_count_180d", "monetary_180d",
    "avg_invoice_value_180d", "units_180d", "distinct_products_180d",
    "active_days_180d",
]
REQUIRED = NUMERIC + ["country"]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


st.set_page_config(page_title="Customer Retention Intelligence", page_icon="🛍️", layout="wide")
st.title("Customer Retention & Revenue Intelligence")
st.caption("UCI Online Retail II · 180-day behavioural history · 90-day repeat-purchase propensity")
model = load_model()
overview, scoring, evaluation = st.tabs(["Overview", "Customer scoring", "Model evaluation"])

with overview:
    a, b, c = st.columns(3)
    a.metric("Feature lookback", "180 days")
    b.metric("Prediction horizon", "90 days")
    c.metric("Scoring unit", "Customer snapshot")
    st.write("Combine transaction-level SQL analysis, customer RFM behaviour and a temporally evaluated propensity model to prioritize retention outreach.")
    st.markdown("[Official UCI dataset documentation](https://archive.ics.uci.edu/dataset/502/online+retail%2Bii)")
    st.warning("This predicts a future repeat purchase. It is not a directly observed contractual churn label or a causal estimate of campaign impact.")

with scoring:
    st.subheader("Score engineered customer snapshots")
    st.write("Upload a CSV containing one row per customer snapshot and all required feature columns.")
    upload = st.file_uploader("Customer feature CSV", type=["csv"])
    if upload is not None:
        frame = pd.read_csv(upload)
        missing = sorted(set(REQUIRED) - set(frame.columns))
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
        elif model is None:
            st.info("Train the model locally with python -m src.train before scoring.")
        else:
            scored = frame.copy()
            scored["repeat_purchase_probability"] = model.predict_proba(frame[REQUIRED])[:, 1]
            scored = scored.sort_values("repeat_purchase_probability", ascending=False)
            capacity = st.slider("Outreach capacity", 1, max(1, len(scored)), min(50, len(scored)))
            st.dataframe(scored.head(capacity), use_container_width=True)
            st.download_button("Download scored file", scored.to_csv(index=False).encode("utf-8"),
                               "customer_retention_scores.csv", "text/csv")

with evaluation:
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        test = metrics.get("test_metrics", {})
        a, b, c, d = st.columns(4)
        a.metric("Selected model", metrics.get("selected_model", "N/A").replace("_", " ").title())
        b.metric("PR-AUC", f"{test['pr_auc']:.3f}" if test.get("pr_auc") is not None else "N/A")
        c.metric("ROC-AUC", f"{test['roc_auc']:.3f}" if test.get("roc_auc") is not None else "N/A")
        d.metric("F1", f"{test.get('f1', 0):.3f}")
        st.json(metrics.get("capacity_metrics", {}))
        st.caption("Historical test metrics; not a current production performance guarantee.")
    else:
        st.info("Evaluation metrics will appear after the local training pipeline runs.")

st.divider()
st.caption("Dataset attribution: Chen, D. (2012), UCI Machine Learning Repository, DOI 10.24432/C5CG6D (CC BY 4.0).")
