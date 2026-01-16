# Heart Failure Mortality Risk Predictor

**Interactive web application** that estimates the probability of death in patients with heart failure using clinical records and machine learning.

Built with **Streamlit** + **Random Forest Classifier**

<p align="center">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
</p>

## 📌 Project Overview

This project aims to help understand and predict mortality risk in heart failure patients based on clinical features.  
A **Random Forest** model is trained on the well-known **Heart Failure Clinical Records Dataset** and deployed as an easy-to-use interactive web application.

Users can input patient clinical values and immediately receive an estimated risk probability along with clear risk interpretation.
</p>

#### **Checkout the dashboard below by clicking on the streamlit link below**

# [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://monopetalous-patria-unpretentiously.ngrok-free.dev/)![GitHub stars](https://img.shields.io/github/stars/MUSBAUDEEN-OPS/heart-failure-prediction?style=social)

## ✨ Key Features

- Real-time mortality risk prediction
- Automatic calculation of clinically meaningful **ratio features**
- Clean and intuitive Streamlit interface
- Educational warnings & general medical guidance (non-diagnostic)
- Full exploratory analysis and model training notebooks

## ⚙️ Why Feature Engineering? (Important!)

Many of the strongest predictors of heart failure outcomes are **not just the raw values**, but the **relationships** between them.

That's why this project includes **ratio-based feature engineering**:

```text
Age / Creatinine Phosphokinase     → A_CP
Age / Ejection Fraction            → A_EF
Creatinine Phosphokinase / Ejection Fraction → CP_EF
Platelets / Serum Creatinine       → P_SC
...and several others
```

These ratios often capture hidden physiological relationships that simple individual values miss, for example:

- How age interacts with cardiac stress markers
- How kidney function (serum creatinine) relates to other blood parameters
- Relative severity between cardiac pumping ability (EF) and inflammation markers (CPK)

These engineered features usually improve model performance and make the prediction more clinically interpretable.

## 🛠️ Tech Stack

- Python 3.8+
- Streamlit – interactive web app
- scikit-learn – Random Forest + preprocessing
- pandas & numpy – data handling
- Feature-engineered ratios for better clinical insight

# ⚠️ Important Disclaimer

**This tool is created for educational and awareness purposes only.**
**It is NOT a medical device, NOT a diagnostic tool, and should never replace professional medical judgment, physical examination, or clinical decision-making.**
**Always consult a qualified cardiologist or healthcare professional.**

## 📄 License
Feel free to use, modify, and learn from this project.

Made with ❤️ by Ibrahim Musbaudeen
Kwara, Nigeria • January 2026
