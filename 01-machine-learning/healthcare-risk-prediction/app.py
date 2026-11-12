"""Educational Streamlit interface for the trained heart-disease classifier."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "healthcare_risk_pipeline.joblib"

st.set_page_config(page_title="Healthcare Risk Prediction Demo", page_icon="🫀", layout="centered")
st.title("Healthcare Risk Prediction")
st.caption("Educational machine-learning demonstration using the UCI Heart Disease dataset.")

st.warning(
    "Research and education only. This application is not a medical device and does not "
    "provide a diagnosis, screening result, or treatment recommendation. Do not enter real "
    "patient information. Consult a qualified healthcare professional for medical concerns."
)

if not MODEL_PATH.exists():
    st.info("No trained model found. From this project folder, run: python -m src.train")
    st.stop()

bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]
feature_columns = bundle["feature_columns"]
threshold = float(bundle.get("threshold", 0.5))

with st.form("prediction_form"):
    st.subheader("Example clinical attributes")
    st.caption("Use synthetic/example values only. Dataset codes are shown in the selectors.")

    col1, col2 = st.columns(2)
    values = {}
    with col1:
        values["age"] = st.number_input("Age (years)", min_value=1, max_value=110, value=54)
        values["sex"] = st.selectbox("Sex code", options=[(1, "Male (1)"), (0, "Female (0)")], format_func=lambda x: x[1])[0]
        values["cp"] = st.selectbox(
            "Chest pain type",
            options=[(1, "Typical angina (1)"), (2, "Atypical angina (2)"), (3, "Non-anginal pain (3)"), (4, "Asymptomatic (4)")],
            format_func=lambda x: x[1],
        )[0]
        values["trestbps"] = st.number_input("Resting blood pressure", min_value=50, max_value=250, value=130)
        values["chol"] = st.number_input("Serum cholesterol", min_value=50, max_value=700, value=240)
        values["fbs"] = st.selectbox("Fasting blood sugar > 120 mg/dl", options=[(0, "No (0)"), (1, "Yes (1)")], format_func=lambda x: x[1])[0]
        values["restecg"] = st.selectbox("Resting ECG code", options=[0, 1, 2])
    with col2:
        values["thalach"] = st.number_input("Maximum heart rate achieved", min_value=50, max_value=250, value=150)
        values["exang"] = st.selectbox("Exercise-induced angina", options=[(0, "No (0)"), (1, "Yes (1)")], format_func=lambda x: x[1])[0]
        values["oldpeak"] = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        values["slope"] = st.selectbox("Peak exercise ST slope code", options=[1, 2, 3])
        values["ca"] = st.selectbox("Major vessels (ca) code", options=[0, 1, 2, 3])
        values["thal"] = st.selectbox("Thalassemia (thal) code", options=[3, 6, 7])

    submitted = st.form_submit_button("Run model demonstration", type="primary")

if submitted:
    input_frame = pd.DataFrame([{column: values[column] for column in feature_columns}])
    probability = float(pipeline.predict_proba(input_frame)[0, 1])
    prediction = int(probability >= threshold)

    st.subheader("Model output")
    st.metric("Model score for the positive dataset label", f"{probability:.1%}")
    if prediction == 1:
        st.error("The model assigned the positive class (dataset label: presence).")
    else:
        st.success("The model assigned the negative class (dataset label: absence).")

    st.caption(
        f"Demonstration threshold: {threshold:.2f}. This score is not a calibrated personal "
        "risk estimate and must not be interpreted as a diagnosis."
    )

st.divider()
st.caption(f"Model selected during training: {bundle.get('selected_model', 'Unknown')}")
