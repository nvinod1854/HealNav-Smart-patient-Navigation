import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import numpy as np

# -------------------------------
# Load model and encoders
# -------------------------------
model = joblib.load("priority_model.pkl")
feature_encoders = joblib.load("feature_encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

FEATURE_COLUMNS = [
    "age",
    "gender",
    "chest_pain",
    "breathlessness",
    "fever",
    "pain_level",
    "symptom_duration_days",
    "existing_disease",
    "severity_level"
]

# -------------------------------
# Safe encoding
# -------------------------------
def safe_encode(df, encoders):
    for col, encoder in encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].apply(
                lambda x: x if x in encoder.classes_ else encoder.classes_[0]
            )
            df[col] = encoder.transform(df[col])
    return df

# -------------------------------
# Session state
# -------------------------------
if "patient_queue" not in st.session_state:
    st.session_state.patient_queue = []

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="HealNav – Smart Care Patient Priority Navigation",
    layout="wide"
)

st.title("🏥 HealNav – Smart Care Patient Priority Navigation")
st.write("AI-assisted triage and patient routing for public hospitals")

st.markdown("---")

left, right = st.columns([1, 1])

# ==================================================
# PATIENT INPUT
# ==================================================
with left:
    st.subheader("👤 Patient Input")

    age = st.slider("Age", 1, 100, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    chest_pain = st.selectbox("Chest Pain", [1, 0], format_func=lambda x: "Yes" if x else "No")
    breathlessness = st.selectbox("Breathlessness", [1, 0], format_func=lambda x: "Yes" if x else "No")
    fever = st.selectbox("Fever", [1, 0], format_func=lambda x: "Yes" if x else "No")
    pain_level = st.selectbox("Pain Level", ["mild", "moderate", "severe"])
    severity_level = st.selectbox("Severity Level", ["Low", "Medium", "High", "Critical"])
    symptom_duration_days = st.slider("Symptom Duration (Days)", 0, 14, 2)
    existing_disease = st.selectbox(
        "Existing Disease",
        ["None", "Diabetes", "Heart Disease", "Asthma", "Hypertension"]
    )

    if st.button("🔍 Predict Priority"):
        input_data = pd.DataFrame([{
            "age": age,
            "gender": gender,
            "chest_pain": chest_pain,
            "breathlessness": breathlessness,
            "fever": fever,
            "pain_level": pain_level,
            "symptom_duration_days": symptom_duration_days,
            "existing_disease": existing_disease,
            "severity_level": severity_level
        }])

        # Ensure all features exist
        for col in FEATURE_COLUMNS:
            if col not in input_data.columns:
                input_data[col] = 0

        input_data = input_data[FEATURE_COLUMNS]

        # Encode categorical columns
        input_data = safe_encode(input_data, feature_encoders)
        # SAFE numeric conversion (never crashes)
        input_data = input_data.apply(pd.to_numeric, errors="coerce")
        input_data = input_data.fillna(0)

       
        X = np.asarray(input_data.values, dtype=float)

        # Predict
        prediction = model.predict(X)
        priority = target_encoder.inverse_transform(prediction)[0]

        if priority == "High":
            st.error("🔴 HIGH PRIORITY – Go to Emergency immediately")
        elif priority == "Medium":
            st.warning("🟡 MEDIUM PRIORITY – Consult doctor today")
        else:
            st.success("🟢 LOW PRIORITY – OPD visit can be delayed")

        # Add to doctor dashboard
        st.session_state.patient_queue.append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Age": age,
            "Gender": gender,
            "Severity": severity_level,
            "Priority": priority
        })

# ==================================================
# DOCTOR DASHBOARD
# ==================================================
with right:
    st.subheader("👨‍⚕️ Doctor Dashboard")

    if st.session_state.patient_queue:
        df = pd.DataFrame(st.session_state.patient_queue)
        priority_map = {"High": 0, "Medium": 1, "Low": 2}
        df["rank"] = df["Priority"].map(priority_map)
        df = df.sort_values("rank").drop(columns="rank")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No patients yet.")

st.markdown("---")
st.caption("⚠️ HealNav assists doctors. It does not replace clinical judgment.")
