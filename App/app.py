# Paste the ENTIRE app.py code from above here
# (You can copy-paste the whole block from the first part)
# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

# ───────────────────────────────────────────────────────────────
# Cache the model training (runs only once)
# ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    # IMPORTANT: Update this path to where your CSV file is located
    # In Colab, after downloading with kagglehub, it's usually in this folder:
    csv_path = "Data/heart_failure_clinical_records_dataset.csv"  
    
    if not os.path.exists(csv_path):
        st.error(f"Dataset not found at: {csv_path}\nPlease make sure the file is in the correct location.")
        st.stop()

    df = pd.read_csv(csv_path)
    
    # Feature engineering - create ratio features
    df["A_CP"]  = df['age'] / df['creatinine_phosphokinase']
    df["A_EF"]  = df['age'] / df['ejection_fraction']
    df["A_P"]   = df['age'] / df['platelets']
    df["A_SC"]  = df['age'] / df['serum_creatinine']
    df["A_SS"]  = df['age'] / df['serum_sodium']
    
    df["CP_EF"] = df['creatinine_phosphokinase'] / df['ejection_fraction']
    df["CP_P"]  = df['creatinine_phosphokinase'] / df['platelets']
    df["CP_SC"] = df['creatinine_phosphokinase'] / df['serum_creatinine']
    df["CP_SS"] = df['creatinine_phosphokinase'] / df['serum_sodium']
    
    df["EF_P"]  = df['ejection_fraction'] / df['platelets']
    df["EF_SC"] = df['ejection_fraction'] / df['serum_creatinine']
    df["EF_SS"] = df['ejection_fraction'] / df['serum_sodium']
    
    df["P_SC"]  = df['platelets'] / df['serum_creatinine']
    df["P_SS"]  = df['platelets'] / df['serum_sodium']
    df["SC_SS"] = df['serum_creatinine'] / df['serum_sodium']

    # Prepare features and target
    X = df.drop("DEATH_EVENT", axis=1)
    y = df["DEATH_EVENT"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2011, stratify=y
    )

    # Define columns
    numerical_cols = [
        "age", "creatinine_phosphokinase", "ejection_fraction", "platelets",
        "serum_creatinine", "serum_sodium", "time",
        "A_CP", "A_EF", "A_P", "A_SC", "A_SS",
        "CP_EF", "CP_P", "CP_SC", "CP_SS", "EF_P", "EF_SC", "EF_SS", "P_SC", "P_SS", "SC_SS"
    ]

    categorical_cols = ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"]

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ]
    )

    # Pipeline
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=2011,
                    class_weight="balanced",
                    max_depth=10,
                    min_samples_split=5,
                ),
            ),
        ]
    )

    # Train
    model.fit(X_train, y_train)

    return model

# ───────────────────────────────────────────────────────────────
# Load the trained model
# ───────────────────────────────────────────────────────────────
model = load_model()

# ───────────────────────────────────────────────────────────────
#                STREAMLIT INTERFACE
# ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Heart Failure Risk Predictor", layout="wide")

st.title("Heart Failure Mortality Risk Predictor")
st.markdown("Enter the patient's clinical values. The model automatically computes important ratios.")

# ─── Input Layout ────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographics & Cardiac")
    age = st.number_input("Age (years)", 18, 100, 65, step=1)
    ejection_fraction = st.number_input("Ejection Fraction (%)", 5, 80, 40, step=1)
    time = st.number_input("Follow-up time (days)", 0, 400, 100, step=1)

with col2:
    st.subheader("Lab Values")
    serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", 0.1, 10.0, 1.2, step=0.1)
    serum_sodium = st.number_input("Serum Sodium (mEq/L)", 110, 150, 136, step=1)
    creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase (mcg/L)", 10, 10000, 250, step=10)
    platelets = st.number_input("Platelets (kiloplatelets/mL)", 20000, 800000, 263000, step=1000)

# Binary factors
st.subheader("Risk Factors")
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    anaemia = st.selectbox("Anaemia", [0, 1], format_func=lambda x: "Yes" if x else "No")
with c2:
    diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "Yes" if x else "No")
with c3:
    hbp = st.selectbox("High BP", [0, 1], format_func=lambda x: "Yes" if x else "No")
with c4:
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
with c5:
    smoking = st.selectbox("Smoking", [0, 1], format_func=lambda x: "Yes" if x else "No")

# ─── Prepare input data ──────────────────────────────────────────
input_dict = {
    "age": age,
    "anaemia": anaemia,
    "creatinine_phosphokinase": creatinine_phosphokinase,
    "diabetes": diabetes,
    "ejection_fraction": ejection_fraction,
    "high_blood_pressure": hbp,
    "platelets": platelets,
    "serum_creatinine": serum_creatinine,
    "serum_sodium": serum_sodium,
    "sex": sex,
    "smoking": smoking,
    "time": time,
}

# Compute engineered features (same as training)
input_dict["A_CP"]   = age / creatinine_phosphokinase if creatinine_phosphokinase != 0 else 0
input_dict["A_EF"]   = age / ejection_fraction if ejection_fraction != 0 else 0
input_dict["A_P"]    = age / platelets if platelets != 0 else 0
input_dict["A_SC"]   = age / serum_creatinine if serum_creatinine != 0 else 0
input_dict["A_SS"]   = age / serum_sodium if serum_sodium != 0 else 0

input_dict["CP_EF"]  = creatinine_phosphokinase / ejection_fraction if ejection_fraction != 0 else 0
input_dict["CP_P"]   = creatinine_phosphokinase / platelets if platelets != 0 else 0
input_dict["CP_SC"]  = creatinine_phosphokinase / serum_creatinine if serum_creatinine != 0 else 0
input_dict["CP_SS"]  = creatinine_phosphokinase / serum_sodium if serum_sodium != 0 else 0

input_dict["EF_P"]   = ejection_fraction / platelets if platelets != 0 else 0
input_dict["EF_SC"]  = ejection_fraction / serum_creatinine if serum_creatinine != 0 else 0
input_dict["EF_SS"]  = ejection_fraction / serum_sodium if serum_sodium != 0 else 0

input_dict["P_SC"]   = platelets / serum_creatinine if serum_creatinine != 0 else 0
input_dict["P_SS"]   = platelets / serum_sodium if serum_sodium != 0 else 0

input_dict["SC_SS"]  = serum_creatinine / serum_sodium if serum_sodium != 0 else 0

# Create DataFrame
input_df = pd.DataFrame([input_dict])

# ─── Prediction ──────────────────────────────────────────────────
if st.button("Calculate Risk", type="primary"):
    try:
        prob_death = model.predict_proba(input_df)[0][1]

        st.markdown("---")

        if prob_death >= 0.50:
            st.error(f"**HIGH RISK DETECTED** – Estimated probability of death event: **{prob_death:.1%}**")

            st.warning("""
            ### ⚠️ Important – Please read carefully
            This is a **machine learning estimation only** — **not a medical diagnosis**.
            """)

            # ── Simple rule-based + template advice ───────────────────────
            st.markdown("""
            **Possible urgent considerations** (general information only):
            - Your combination of clinical parameters suggests significantly increased risk
            - Parameters like low ejection fraction, high serum creatinine, anaemia, and/or older age often contribute to higher risk profiles
            - Recent changes in symptoms (worsening shortness of breath, swelling, fatigue, reduced exercise tolerance) would be especially concerning
            
            **Recommended immediate actions** (general guidance):
            1. Contact your cardiologist / heart failure specialist **as soon as possible**
            2. Consider visiting the nearest emergency department if you experience:
               • Severe shortness of breath at rest
               • Sudden significant swelling
               • Chest pain / pressure
               • Fainting / severe dizziness
               • Very rapid or irregular heartbeat with symptoms
            """)

            # ── Optional: More personalized-looking message using basic input ──
            risk_message = "higher" if ejection_fraction < 30 or serum_creatinine > 2.0 else "elevated"
            
            st.info(f"""
            Given your reported ejection fraction of {ejection_fraction}% 
            and serum creatinine of {serum_creatinine} mg/dL, 
            this represents a **{risk_message}** risk category according to many clinical studies.
            
            **Strongly recommended**: Discuss these results and your current symptoms 
            with a qualified cardiologist in the **next few days** at the latest.
            """)

        else:
            st.success(f"**LOWER ESTIMATED RISK** – Probability of death event: **{prob_death:.1%}**")
            st.info("""
            This is a **favorable** result according to the model.
            Continue following your current treatment plan and regular follow-ups.
            """)

        st.caption("""
        ⚕️ This tool is for educational & awareness purposes only.  
        It cannot replace professional medical evaluation, physical examination, 
        laboratory tests, imaging studies, or a physician's judgment.
        """)

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
