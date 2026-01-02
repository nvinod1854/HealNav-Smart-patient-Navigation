import streamlit as st
import pandas as pd
import joblib

# Load trained model and encoders
model = joblib.load("priority_model.pkl")
feature_encoders = joblib.load("feature_encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

st.set_page_config(page_title="Smart Care Triage", layout="centered")

st.title("🏥 Smart Care Patient Triage System")
st.write("AI-assisted patient priority prediction for public hospitals")

st.markdown("---")

age = st.slider("Age", 1, 100, 30)
gender = st.selectbox("Gender", ["Male", "Female"])

chest_pain = st.selectbox("Chest Pain", ["Yes", "No"])
breathlessness = st.selectbox("Breathlessness", ["Yes", "No"])
fever = st.selectbox("Fever", ["Yes", "No"])

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

    for col, encoder in feature_encoders.items():
        input_data[col] = encoder.transform(input_data[col])

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
        st.write("Non-urgent. OPD visit can be scheduled later.")

    st.caption("⚠️ This system assists doctors and does not replace medical judgment.")
