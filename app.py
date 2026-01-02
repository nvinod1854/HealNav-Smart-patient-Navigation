import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load model and encoder
# -------------------------------
model = joblib.load("priority_model.pkl")
target_encoder = joblib.load("target_encoder.pkl")

st.set_page_config(page_title="Smart Care Patient Triage", layout="centered")

st.title("🏥 Smart Care Patient Triage System")
st.write("AI-assisted patient priority prediction for public hospitals")

st.markdown("---")

# -------------------------------
# Manual mappings (IMPORTANT)
# -------------------------------
gender_map = {"Male": 1, "Female": 0}

pain_level_map = {
    "mild": 0,
    "moderate": 1,
    "severe": 2
}

severity_level_map = {
    "Low": 0,
    "Medium": 1,
    "High": 2,
    "Critical": 3
}

existing_disease_map = {
    "None": 0,
    "Diabetes": 1,
    "Heart Disease": 2,
    "Asthma": 3,
    "Hypertension": 4
}

# -------------------------------
# User Inputs
# -------------------------------
age = st.slider("Age", 1, 100, 30)

gender = st.selectbox("Gender", ["Male", "Female"])
chest_pain = st.selectbox("Chest Pain", ["No", "Yes"])
breathlessness = st.selectbox("Breathlessness", ["No", "Yes"])
fever = st.selectbox("Fever", ["No", "Yes"])

pain_level = st.selectbox("Pain Level", ["mild", "moderate", "severe"])
severity_level = st.selectbox("Severity Level", ["Low", "Medium", "High", "Critical"])

symptom_duration_days = st.slider("Symptom Duration (Days)", 0, 14, 2)

existing_disease = st.selectbox(
    "Existing Disease",
    ["None", "Diabetes", "Heart Disease", "Asthma", "Hypertension"]
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict Patient Priority"):

    input_data = pd.DataFrame([[
        age,
        gender_map[gender],
        1 if chest_pain == "Yes" else 0,
        1 if breathlessness == "Yes" else 0,
        1 if fever == "Yes" else 0,
        pain_level_map[pain_level],
        symptom_duration_days,
        existing_disease_map[existing_disease],
        severity_level_map[severity_level]
    ]], columns=[
        "age",
        "gender",
        "chest_pain",
        "breathlessness",
        "fever",
        "pain_level",
        "symptom_duration_days",
        "existing_disease",
        "severity_level"
    ])

    # Ensure correct feature order
    input_data = input_data[model.feature_names_in_]

    prediction = model.predict(input_data)
    priority = target_encoder.inverse_transform(prediction)[0]

    st.markdown("---")

    if priority == "High":
        st.error("🔴 HIGH PRIORITY")
        st.write("Immediate medical attention required. Proceed to Emergency.")

    elif priority == "Medium":
        st.warning("🟡 MEDIUM PRIORITY")
        st.write("Doctor consultation recommended today.")

    else:
        st.success("🟢 LOW PRIORITY")
        st.write("Non-urgent case. OPD visit can be scheduled later.")

    st.caption("⚠️ This system assists doctors and does not replace medical judgment.")
