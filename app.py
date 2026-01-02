import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load model and encoders
# -------------------------------
model = joblib.load("priority_model.pkl")
feature_encoders = joblib.load("feature_encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Smart Care Patient Triage",
    layout="centered"
)

st.title("🏥 Smart Care Patient Triage System")
st.write("AI-assisted patient priority prediction for public hospitals")

st.markdown("---")

# -------------------------------
# User Inputs
# -------------------------------
age = st.slider("Age", 1, 100, 30)

gender = st.selectbox("Gender", ["Male", "Female"])

# IMPORTANT: use numeric values for Yes/No
chest_pain = st.selectbox(
    "Chest Pain",
    [1, 0],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

breathlessness = st.selectbox(
    "Breathlessness",
    [1, 0],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

fever = st.selectbox(
    "Fever",
    [1, 0],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

pain_level = st.selectbox(
    "Pain Level",
    ["mild", "moderate", "severe"]
)

severity_level = st.selectbox(
    "Severity Level",
    ["Low", "Medium", "High", "Critical"]
)

symptom_duration_days = st.slider(
    "Symptom Duration (Days)",
    0, 14, 2
)

existing_disease = st.selectbox(
    "Existing Disease",
    ["None", "Diabetes", "Heart Disease", "Asthma", "Hypertension"]
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict Patient Priority"):

    # Create DataFrame
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

    # Encode categorical columns ONLY
    for col, encoder in feature_encoders.items():
        if col in input_data.columns:
            input_data[col] = encoder.transform(input_data[col])

    # Ensure column order matches training
    input_data = input_data[model.feature_names_in_]

    # Predict
    prediction = model.predict(input_data)
    priority = target_encoder.inverse_transform(prediction)[0]

    st.markdown("---")

    # -------------------------------
    # Output
    # -------------------------------
    if priority == "High":
        st.error("🔴 HIGH PRIORITY")
        st.write("Immediate medical attention required. Proceed to Emergency Department.")

    elif priority == "Medium":
        st.warning("🟡 MEDIUM PRIORITY")
        st.write("Doctor consultation recommended today.")

    else:
        st.success("🟢 LOW PRIORITY")
        st.write("Non-urgent case. OPD visit can be scheduled later.")

    st.caption(
        "⚠️ This system assists in patient prioritization and does not replace doctors."
    )
