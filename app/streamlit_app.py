"""
streamlit_app.py
Interactive web interface for the Diabetes Prediction model.

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
import os
import subprocess

# Repo root is one level up from app/
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))

import numpy as np
import joblib
import streamlit as st

MODEL_PATH  = os.path.join(REPO_ROOT, 'models', 'best_model.pkl')
SCALER_PATH = os.path.join(REPO_ROOT, 'models', 'scaler.pkl')
DATA_PATH   = os.path.join(REPO_ROOT, 'data', 'diabetes.csv')
TRAIN_SCRIPT = os.path.join(REPO_ROOT, 'src', 'train_model.py')

FEATURE_ORDER = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]


def models_exist() -> bool:
    return os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)


def auto_train():
    """Run train_model.py from the repo root so all relative paths resolve correctly."""
    os.makedirs(os.path.join(REPO_ROOT, 'models'), exist_ok=True)
    os.makedirs(os.path.join(REPO_ROOT, 'images'), exist_ok=True)
    result = subprocess.run(
        [sys.executable, TRAIN_SCRIPT],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True
    )
    return result.returncode == 0, result.stdout, result.stderr


@st.cache_resource
def load_artifacts():
    """Load model and scaler once and cache them across sessions."""
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def configure_page():
    st.set_page_config(
        page_title="Diabetes Prediction",
        page_icon="🩺",
        layout="centered",
    )


def render_header():
    st.title("🩺 Diabetes Risk Prediction")
    st.markdown(
        "Enter the patient's clinical measurements below. "
        "The model will estimate the probability of diabetes using a "
        "**Random Forest classifier** trained on the Pima Indians Diabetes Dataset."
    )
    st.divider()


def render_input_form():
    st.subheader("Patient Clinical Data")
    col1, col2 = st.columns(2)

    with col1:
        pregnancies     = st.number_input("Pregnancies", min_value=0, max_value=20, value=3,
                                          help="Number of times the patient has been pregnant")
        glucose         = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=117,
                                          help="Plasma glucose concentration from a 2-hour oral glucose tolerance test")
        blood_pressure  = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=150, value=72,
                                          help="Diastolic blood pressure")
        skin_thickness  = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=23,
                                          help="Triceps skin fold thickness")

    with col2:
        insulin = st.number_input("Insulin (μU/mL)", min_value=0, max_value=900, value=30,
                                  help="2-hour serum insulin")
        bmi     = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, value=32.0, step=0.1,
                                  help="Body Mass Index")
        dpf     = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0,
                                  value=0.37, step=0.001, format="%.3f",
                                  help="Likelihood of diabetes based on family history")
        age     = st.number_input("Age (years)", min_value=18, max_value=120, value=29)

    return [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]


def render_prediction(model, scaler, feature_values: list):
    st.divider()
    features        = np.array(feature_values).reshape(1, -1)
    features_scaled = scaler.transform(features)

    prediction  = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    if prediction == 1:
        st.error("### ⚠️ High Diabetes Risk Detected")
        st.metric(label="Predicted Probability of Diabetes", value=f"{probability * 100:.1f}%")
        st.warning(
            "This model indicates a **high risk** of diabetes. "
            "This is not a medical diagnosis. Please consult a qualified healthcare professional."
        )
    else:
        st.success("### ✅ Low Diabetes Risk Detected")
        st.metric(label="Predicted Probability of Diabetes", value=f"{probability * 100:.1f}%")
        st.info(
            "This model indicates a **low risk** of diabetes. "
            "Continue maintaining a healthy diet and active lifestyle."
        )

    st.progress(float(probability), text=f"Risk Score: {probability:.2%}")


def render_footer():
    st.divider()
    st.caption(
        "**Disclaimer:** This tool is for educational purposes only and is not a substitute "
        "for professional medical advice, diagnosis, or treatment. Model accuracy: ~82% on test data."
    )
    st.caption("Model: Random Forest | Dataset: Pima Indians Diabetes | By Yash")


def main():
    configure_page()
    render_header()

    # Auto-train if model files are missing (e.g. first run on Streamlit Cloud)
    if not models_exist():
        if not os.path.exists(DATA_PATH):
            st.error(
                "Dataset not found at `data/diabetes.csv`. "
                "Please ensure it is committed to the repository."
            )
            st.stop()

        with st.spinner("⏳ First-time setup: training model on the dataset (takes ~30 seconds)..."):
            success, stdout, stderr = auto_train()

        if not success:
            st.error("Model training failed. Check the logs below.")
            st.code(stderr)
            st.stop()

        st.success("Model trained and ready!")
        st.cache_resource.clear()  # Force reload of newly saved model

    model, scaler = load_artifacts()

    feature_values = render_input_form()

    st.divider()
    if st.button("🔍 Predict Diabetes Risk", use_container_width=True, type="primary"):
        render_prediction(model, scaler, feature_values)

    render_footer()


if __name__ == "__main__":
    main()
