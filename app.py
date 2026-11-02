from pathlib import Path
import pickle

import numpy as np
import streamlit as st

# Resolve the model relative to this file so the app works from any launch directory.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "diabetes_model.pkl"

st.set_page_config(
    page_title="AI Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1576091160399-112ba8d25d1f");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}
h1, h2, h3 {
    color: white;
    text-align: center;
}
label {
    color: white !important;
    font-weight: bold;
}
[data-testid="stMetricValue"] {
    color: #00ff99;
}
</style>
""", unsafe_allow_html=True)

if not MODEL_PATH.exists():
    st.error("Model file not found. Ensure diabetes_model.pkl is in the project root.")
    st.stop()

with MODEL_PATH.open("rb") as model_file:
    model = pickle.load(model_file)

st.sidebar.title("🩺 Project Information")
st.sidebar.info("""
AI-Based Diabetes Prediction System

Algorithm:
• Random Forest Classifier

Dataset:
• Pima Indians Diabetes Dataset

Purpose:
• Educational diabetes-risk estimation
""")

st.markdown(
    "<h1>🩺 AI-Based Diabetes Prediction System</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h4 style='text-align:center;color:white;'>Estimate diabetes risk using machine learning</h4>",
    unsafe_allow_html=True
)

st.warning(
    "For educational use only. This result is not a medical diagnosis "
    "and must not replace advice or testing from a qualified healthcare professional."
)

preg = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose", min_value=0, max_value=300, value=100)
bp = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=1, max_value=120, value=30)

if st.button("🔍 Estimate Diabetes Risk"):
    input_data = np.array([[
        preg, glucose, bp, skin, insulin, bmi, dpf, age
    ]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)
    risk = float(probability[0][1] * 100)

    st.subheader("📊 Risk Estimate")
    st.progress(int(round(risk)))
    st.metric("Model-estimated probability for the positive class", f"{risk:.2f}%")

    if prediction == 1:
        st.error(f"The model classified this input in the higher-risk class ({risk:.2f}% estimated probability).")
        st.info("Please consult a qualified healthcare professional for proper assessment.")
    else:
        st.success(f"The model classified this input in the lower-risk class ({risk:.2f}% estimated probability).")
        st.info("A lower model estimate does not rule out diabetes. Consult a healthcare professional if concerned.")
