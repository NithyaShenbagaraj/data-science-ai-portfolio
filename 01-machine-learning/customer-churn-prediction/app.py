"""Streamlit batch-scoring demo for customer churn probabilities."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from src.features import engineer_features

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "customer_churn_pipeline.joblib"
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="wide")
st.title("Customer Churn Prediction")
st.caption("Educational machine-learning demonstration — not a guarantee of customer behaviour.")

if not MODEL_PATH.exists():
    st.warning("The trained model is not available yet. Download the dataset and run python -m src.train first.")
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()
uploaded = st.file_uploader("Upload customer data (CSV)", type=["csv"])
if uploaded is not None:
    raw = pd.read_csv(uploaded)
    features = engineer_features(raw)
    probabilities = model.predict_proba(features)[:, 1]
    results = raw.copy()
    results["churn_probability"] = probabilities
    threshold = st.slider("Churn-priority threshold", 0.05, 0.95, 0.50, 0.05)
    results["retention_priority"] = results["churn_probability"].ge(threshold).map(
        {True: "Prioritize", False: "Standard"}
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Customers scored", f"{len(results):,}")
    col2.metric("High-priority customers", f"{(probabilities >= threshold).sum():,}")
    col3.metric("Average churn probability", f"{probabilities.mean():.1%}")
    st.dataframe(results.sort_values("churn_probability", ascending=False), use_container_width=True)
    st.download_button(
        "Download scored customers",
        results.to_csv(index=False).encode("utf-8"),
        file_name="customer_churn_predictions.csv",
        mime="text/csv",
    )
    st.info("Use predictions to support review and retention planning. Do not treat probabilities as certainty or upload identifiable customer data to a public demo.")
else:
    st.info("Upload a CSV with the same input columns used by the training dataset. The target column and customer ID are optional.")
