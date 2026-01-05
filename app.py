from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load model & encoders
model = joblib.load("priority_model.pkl")
feature_encoders = joblib.load("feature_encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

@app.route("/predict", methods=["POST"])
def predict_priority():
    data = request.json

    # Create DataFrame
    input_data = pd.DataFrame([{
        "age": data["age"],
        "gender": data["gender"],
        "chest_pain": data["chest_pain"],
        "breathlessness": data["breathlessness"],
        "fever": data["fever"],
        "pain_level": data["pain_level"],
        "symptom_duration_days": data["symptom_duration_days"],
        "existing_disease": data["existing_disease"],
        "severity_level": data["severity_level"]
    }])

    # Encode categorical features
    for col, encoder in feature_encoders.items():
        if col in input_data.columns:
            input_data[col] = encoder.transform(input_data[col])

    # Ensure correct order
    input_data = input_data[model.feature_names_in_]

    # Predict
    prediction = model.predict(input_data)
    priority = target_encoder.inverse_transform(prediction)[0]

    return jsonify({"priority_level": priority})

if __name__ == "__main__":
    app.run(debug=True)
