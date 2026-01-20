import streamlit as st
import pandas as pd
import joblib
from PIL import Image  # <-- Added PIL

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)


# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* App background */
.stApp {
    background-color: #0f172a;
}

/* Main container */
.block-container {
    background-color: #1e293b;
    padding: 2.5rem 3rem;
    border-radius: 16px;
    max-width: 1000px;
    margin-top: 2rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
}

/* Titles */
h1 {
    color: #f43f5e;
    font-weight: 800;
}

h2, h3 {
    color: #e5e7eb;
    font-weight: 700;
}

/* Labels */
label {
    color: #e5e7eb !important;
    font-weight: 600;
}

/* Input fields */
input, textarea, select {
    background-color: #f9fafb !important;
    color: #111827 !important;
    border-radius: 8px;
}

/* Dropdown selected text */
div[data-baseweb="select"] > div {
    background-color: #f9fafb !important;
    color: #111827 !important;
}

/* Number input +/- buttons */
button[data-baseweb="button"] {
    color: #111827 !important;
}

/* Sliders */
span {
    color: #e5e7eb;
}

/* Predict button */
.stButton > button {
    background-color: #f43f5e;
    color: white !important;
    font-size: 18px;
    height: 3em;
    border-radius: 10px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("model/KNN_heart.pkl")
scaler = joblib.load("model/scaler.pkl")
expected_columns = joblib.load("model/columns.pkl")

# ---------------- HEADER ----------------
st.title("❤️ Heart Disease Risk Prediction")

# Load and resize image with Pillow
img = Image.open("assets/heart_bg.jpg")

# Resize: wider width while keeping original height
new_width =2000
img = img.resize((new_width, img.height))

st.image(img, use_container_width=False)  # keeps the resized dimensions

st.markdown("### **This tool estimates heart disease risk using ML.**")
st.divider()

# ---------------- INPUT SECTION ----------------
st.subheader("🩺 Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)

with col2:
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

st.divider()

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Heart Disease Risk", use_container_width=True):

    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]

    st.divider()

    if prediction == 1:
        st.error("⚠️ **High Risk of Heart Disease**")
        st.markdown(
            "🔴 Please consult a qualified medical professional for further evaluation."
        )
    else:
        st.success("✅ **Low Risk of Heart Disease**")
        st.markdown(
            "🟢 Maintain a healthy lifestyle and regular medical checkups."
        )

# ---------------- FOOTER ----------------
st.divider()
st.caption(
    "⚠️ Disclaimer: This application provides an estimate based on a machine learning model and "
    "should not be used as a substitute for professional medical advice."
)
