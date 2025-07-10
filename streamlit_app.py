# streamlit_app.py

import streamlit as st
import numpy as np
import joblib
import json

# Load model, scaler, and column names
model = joblib.load("Random Forest_spam_model.pkl")  # change name if best model is different
scaler = joblib.load("scaler.pkl")
with open("feature_columns.json", "r") as f:
    columns = json.load(f)

# App Title
st.title("📧 Spam Email Classifier")
st.markdown("Enter email features below to predict whether it's **Spam** or **Not Spam**.")

# Create input form
input_data = []
for col in columns:
    val = st.number_input(f"{col}", value=0.0, step=0.1)
    input_data.append(val)

# Predict button
if st.button("Predict"):
    input_array = np.array([input_data])
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][int(prediction)]

    if prediction == 1:
        st.error(f"🚫 This email is likely **SPAM** with probability {prob:.2%}")
    else:
        st.success(f"✅ This email is **NOT SPAM** with probability {prob:.2%}")
