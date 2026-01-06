import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# -------------------------------
# Load model and encoders
# -------------------------------
model = joblib.load("priority_model.pkl")
feature_encoders = joblib.load("feature_encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

# -------------------------------
# Safe encoding function
# -------------------------------
def safe_encode(input_df, encoders):
    for col, encoder in encoders.items():
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(str)
            input_df[col] = input_df[col].apply(
                lambda x: x if x in encoder.classes_ else encoder.classes_[0]
            )
            input_df[col] = encoder.transform(input_df[col])
    return input_df

# -------------------------------
# Session state for Doctor Dashboard
# -------------------------------
if "patient_queue" not in st.session_state:
    st.session_state.patient_queue = []

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="HealNav – Smart Care Patient Triage",
    layout="wide"
)

st.title("🏥 HealNav – Smart Care Patient Priority Navigation")
st.write("AI-assisted patient triage and priority routing for public hospitals")

st.markdown("---")

# -------------------------------
# Layout: Patient | Doctor
# -------------------------------
patient_col, doctor_col = st.columns([1, 1])

# ==================================================
# PATIENT SIDE
# ==================================================
with patient_col:
    st.subheader("👤 Patient Input")

    age = st.slider("Age", 1, 100, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])

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

    pain_level = st.selectbox("Pain Level", ["mild", "moderate", "severe"])
    severity_level = st.selectbox("Severity Level", ["Low", "Medium", "High", "Critical"])

    symptom_duration_days = st.slider("Symptom Duration (Days)", 0, 14, 2)

    existing_disease = st.selectbox(
        "Existing Disease",
        ["None", "Diabetes", "Heart Disease", "Asthma", "Hypertension"]
    )

    if st.button("🔍 Predict Patient Priority"):

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

        # Encode safely
        input_data = safe_encode(input_data, feature_encoders)

        # Match training order
        input_data = input_data[model.feature_names_in_]

        # Convert to numeric
        input_data = input_data.astype(float)

        # Predict
        prediction = model.predict(input_data)
        priority = target_encoder.inverse_transform(prediction)[0]

        st.markdown("---")

        # Priority message
        if priority == "High":
            st.error("🔴 HIGH PRIORITY – Go to Emergency Department immediately.")
        elif priority == "Medium":
            st.warning("🟡 MEDIUM PRIORITY – Doctor consultation required today.")
        else:
            st.success("🟢 LOW PRIORITY – OPD visit can be scheduled later.")

        # Add patient to doctor queue
        st.session_state.patient_queue.append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Age": age,
            "Gender": gender,
            "Symptoms": f"CP:{chest_pain}, BL:{breathlessness}, Fever:{fever}",
            "Severity": severity_level,
            "Priority": priority
        })

# ==================================================
# DOCTOR DASHBOARD
# ==================================================
with doctor_col:
    st.subheader("👨‍⚕️ Doctor Dashboard")

    if st.session_state.patient_queue:
        df = pd.DataFrame(st.session_state.patient_queue)

        # Sort by priority
        priority_order = {"High": 0, "Medium": 1, "Low": 2}
        df["priority_rank"] = df["Priority"].map(priority_order)
        df = df.sort_values("priority_rank").drop(columns="priority_rank")

        st.dataframe(df, use_container_width=True)

        st.caption("Patients are automatically sorted by priority")
    else:
        st.info("No patients in queue yet.")

st.markdown("---")

# -------------------------------
# System Explanation
# -------------------------------
st.info(
    """
    **How HealNav Works**
    
    • Patient enters symptoms  
    • AI predicts priority level  
    • Patient receives alert  
    • Doctor dashboard shows prioritized cases  
    """
)

st.caption("⚠️ HealNav assists doctors; it does not replace medical judgment.")
