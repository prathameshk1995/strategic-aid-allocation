import streamlit as st
import joblib
import pandas as pd
import os

# app/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# project root  
ROOT_DIR = os.path.dirname(BASE_DIR) 

model_path = os.path.join(ROOT_DIR, "model", "kmeans_model.pkl") 
scaler_path = os.path.join(ROOT_DIR, "model", "scaler.pkl") 
features_path = os.path.join(ROOT_DIR, "model", "features.pkl") 
thresholds_path = os.path.join(ROOT_DIR, "model", "thresholds.pkl") 

# Load artifacts 
model = joblib.load(model_path) 
scaler = joblib.load(scaler_path) 
features = joblib.load(features_path) 
thresholds = joblib.load(thresholds_path)

st.title("🌍 Strategic Aid Allocation")

st.write("Enter country details to predict aid requirement")

# Inputs
child_mort = st.number_input("Child Mortality")
exports = st.number_input("Exports")
health = st.number_input("Health Spending")
imports = st.number_input("Imports")
income = st.number_input("Income")
inflation = st.number_input("Inflation")
life_expec = st.number_input("Life Expectancy")
total_fer = st.number_input("Total Fertility")
gdpp = st.number_input("GDP per capita")

if st.button("Predict"):

    input_data = [child_mort, exports, health, imports, income,
                  inflation, life_expec, total_fer, gdpp]

    df_input = pd.DataFrame([input_data], columns=features)

    scaled = scaler.transform(df_input)
    prediction = model.predict(scaled)[0]

    # Category
    if prediction == 1:
        category = "🔴 High Risk"
    elif prediction == 2:
        category = "🟡 Medium Risk"
    else:
        category = "🟢 Low Risk"

    # Reasons
    reasons = []

    if child_mort > thresholds['child_mort']:
        reasons.append("High child mortality")

    if income < thresholds['income']:
        reasons.append("Low income")

    if life_expec < thresholds['life_expec']:
        reasons.append("Low life expectancy")

    if inflation > thresholds['inflation']:
        reasons.append("High inflation")

    st.subheader(f"Prediction: {category}")
    st.write("Reasons:", reasons)
