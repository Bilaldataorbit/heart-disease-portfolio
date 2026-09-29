import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide"
)

@st.cache_resource
def load_model():
    model_path = "models/heart_disease_model.pkl"
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

model = load_model()

st.title("🫀 Heart Disease Prediction")
st.markdown(
    """
    This app predicts the likelihood of heart disease based on clinical parameters.
    The model is a **Logistic Regression** trained on the UCI Heart Disease dataset
    with a **ROC-AUC of 0.925**.
    """
)

st.markdown("---")

st.sidebar.header("🧾 Patient Information")
st.sidebar.markdown("Enter the clinical parameters below.")

age = st.sidebar.slider("Age", 20, 90, 55)
sex = st.sidebar.selectbox("Sex", options=[0, 1],
                            format_func=lambda x: "Female" if x == 0 else "Male")

cp = st.sidebar.selectbox(
    "Chest Pain Type",
    options=[0, 1, 2, 3],
    format_func=lambda x: {
        0: "0 - Typical Angina",
        1: "1 - Atypical Angina",
        2: "2 - Non-anginal Pain",
        3: "3 - Asymptomatic"
    }[x]
)

trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 200, 130)
chol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 600, 240)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl",
                            options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

restecg = st.sidebar.selectbox(
    "Resting ECG",
    options=[0, 1, 2],
    format_func=lambda x: {
        0: "0 - Normal",
        1: "1 - ST-T Abnormality",
        2: "2 - LV Hypertrophy"
    }[x]
)

thalach = st.sidebar.slider("Max Heart Rate Achieved", 60, 220, 150)
exang = st.sidebar.selectbox("Exercise-Induced Angina",
                              options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
oldpeak = st.sidebar.slider("ST Depression (oldpeak)", 0.0, 7.0, 1.0, step=0.1)

slope = st.sidebar.selectbox(
    "Slope of Peak Exercise ST Segment",
    options=[0, 1, 2],
    format_func=lambda x: {
        0: "0 - Upsloping",
        1: "1 - Flat",
        2: "2 - Downsloping"
    }[x]
)

ca = st.sidebar.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])

thal = st.sidebar.selectbox(
    "Thalassemia",
    options=[0, 1, 2, 3],
    format_func=lambda x: {
        0: "0 - Unknown",
        1: "1 - Normal",
        2: "2 - Fixed Defect",
        3: "3 - Reversible Defect"
    }[x]
)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Input Summary")
    input_df = pd.DataFrame([{
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg,
        "thalach": thalach, "exang": exang, "oldpeak": oldpeak,
        "slope": slope, "ca": ca, "thal": thal
    }])
    st.dataframe(input_df.T.rename(columns={0: "Value"}))

with col2:
    st.subheader("🎯 Prediction")

    if model is None:
        st.error("❌ Model file not found. Please run the modeling notebook first.")
    else:
        if st.button("Predict", type="primary", use_container_width=True):
            proba = model.predict_proba(input_df)[0][1]
            pred = int(proba >= 0.5)

            if pred == 1:
                st.error(f"⚠️ **High Risk of Heart Disease**")
            else:
                st.success(f"✅ **Low Risk of Heart Disease**")

            st.metric("Disease Probability", f"{proba * 100:.2f}%")
            st.progress(min(proba, 1.0))

            if proba < 0.3:
                st.info("Low risk. Maintain a healthy lifestyle.")
            elif proba < 0.6:
                st.warning("Moderate risk. Consider consulting a cardiologist.")
            else:
                st.error("High risk. Please consult a cardiologist soon.")

st.markdown("---")
st.caption(
    "⚠️ **Disclaimer:** This app is for educational purposes only and is not a medical diagnostic tool. "
    "Always consult a qualified healthcare provider."
)

st.markdown("Built with ❤️ using **Streamlit**, **scikit-learn**, and **SHAP**.")