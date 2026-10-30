"""
Diabetes risk predictor - Streamlit app for the logistic regression assignment.

Run locally:   streamlit run app.py
Needs (same folder): diabetes_model.joblib  (saved by the notebook)
                     diabetes.csv           (only used as a fallback, see below)
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
MODEL_PATH = HERE / "diabetes_model.joblib"
DATA_PATH = HERE / "diabetes.csv"

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]


@st.cache_resource
def load_model():
    """Load the pipeline saved by the notebook (imputer + scaler + logistic regression)."""
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        # Fallback: if the .joblib file is missing or was saved with a different
        # scikit-learn version, rebuild the same model from the CSV.
        df = pd.read_csv(DATA_PATH)
        zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        df[zero_as_missing] = df[zero_as_missing].replace(0, float("nan"))
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=1000)),
        ])
        return model.fit(df[FEATURES], df["Outcome"])


model = load_model()

# ---------------------------------------------------------------- page layout
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺")
st.title("🩺 Diabetes Risk Predictor")
st.write(
    "Enter a patient's measurements to get the model's estimated probability of "
    "diabetes. The model is a logistic regression trained on the Pima Indians "
    "diabetes dataset."
)

# ---------------------------------------------------------------- user inputs
st.subheader("Patient measurements")
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20, 2, step=1)
    glucose = st.number_input("Glucose (mg/dL)", 40, 250, 120, step=1)
    blood_pressure = st.number_input("Diastolic blood pressure (mm Hg)", 30, 130, 72, step=1)
    skin_thickness = st.number_input("Skin thickness (mm)", 5, 100, 29, step=1)

with col2:
    insulin = st.number_input("Insulin (mu U/ml)", 10, 900, 125, step=5)
    bmi = st.number_input("BMI", 15.0, 70.0, 32.0, step=0.1, format="%.1f")
    dpf = st.number_input("Diabetes pedigree function", 0.05, 2.5, 0.40, step=0.01, format="%.2f",
                          help="A score of family history of diabetes. The dataset average is about 0.47.")
    age = st.number_input("Age (years)", 18, 90, 30, step=1)

threshold = st.slider(
    "Decision threshold", 0.10, 0.90, 0.50, 0.05,
    help="Probability above which the app flags the patient as 'high risk'. "
         "Lowering it catches more real cases but gives more false alarms.",
)

# ---------------------------------------------------------------- prediction
if st.button("Predict", type="primary"):
    patient = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                             insulin, bmi, dpf, age]], columns=FEATURES)

    prob = float(model.predict_proba(patient)[0, 1])

    st.subheader("Result")
    st.metric("Estimated probability of diabetes", f"{prob:.1%}")
    st.progress(min(max(prob, 0.0), 1.0))

    if prob >= threshold:
        st.error(f"⚠️ Higher risk: probability is above the {threshold:.0%} threshold.")
    else:
        st.success(f"✅ Lower risk: probability is below the {threshold:.0%} threshold.")

    # Show which inputs pushed the prediction up or down (standardised coefficient x standardised value)
    imputed = model.named_steps["imputer"].transform(patient)
    scaled = model.named_steps["scaler"].transform(imputed)[0]
    coefs = model.named_steps["logreg"].coef_[0]
    contrib = pd.Series(coefs * scaled, index=FEATURES).sort_values()
    with st.expander("What drove this prediction?"):
        st.write(
            "Each bar is that feature's push on the log-odds of diabetes, relative to an "
            "average patient. Positive values raise the risk, negative values lower it."
        )
        st.bar_chart(contrib)

st.caption(
    "⚠️ This is a student project for learning purposes and is **not** a medical "
    "diagnostic tool. Please consult a healthcare professional for medical advice."
)
