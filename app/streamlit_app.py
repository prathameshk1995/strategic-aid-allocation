import streamlit as st
import joblib
import pandas as pd
import os
import numpy as np

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

# 🔥 Step 1: Log Transform
def log_transform(df):
    cols_to_log = ['exports', 'income', 'imports', 'gdpp']
    for col in cols_to_log:
        df[col] = np.log1p(df[col])
    return df

# 🔥 Step 2: IQR Clipping
def clip_inflation(df, thresholds):
    lower = thresholds['inflation_lower']
    upper = thresholds['inflation_upper']
    df['inflation'] = df['inflation'].clip(lower, upper)
    return df

# 🔥 Step 3: Feature Engineering
def create_features(df, thresholds):
    df['high_child_mort'] = (df['child_mort'] > thresholds['child_mort_75']).astype(int)
    df['low_income'] = (df['income'] < thresholds['income_25']).astype(int)
    df['high_inflation'] = (df['inflation'] > thresholds['inflation_75']).astype(int)
    df['low_life_expec'] = (df['life_expec'] < thresholds['life_expec_25']).astype(int)
    return df

# Streamlit UI
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
    
    raw_features = ['child_mort','exports','health','imports','income', 'inflation','life_expec','total_fer','gdpp']

    df_input = pd.DataFrame([input_data], columns=raw_features)

    # ✅ Step 1: Log Transform
    df_input = log_transform(df_input)

    # ✅ Step 2: Clip inflation
    df_input = clip_inflation(df_input, thresholds)

    # ✅ Step 3: Feature Engineering
    df_input = create_features(df_input, thresholds)

    # ✅ Step 4: Align columns
    df_input = df_input[features]

    # ✅ Step 5: Scale
    scaled = scaler.transform(df_input)

    # ✅ Step 6: Predict
    prediction = model.predict(scaled)[0]

    # Category
    cluster_map = {
        1: "🔴 High Risk - Needs Immediate Aid",
        2: "🟡 Medium Risk - Economic Instability",
        0: "🟢 Low Risk - Stable"
    }

    category = cluster_map.get(prediction, "Unknown")

    # Reasons
    reasons = []

    if child_mort > thresholds['child_mort_75']:
        reasons.append("High child mortality")

    if income < thresholds['income_25']:
        reasons.append("Low income")

    if life_expec < thresholds['life_expec_25']:
        reasons.append("Low life expectancy")

    if inflation > thresholds['inflation_75']:
        reasons.append("High inflation")

    if len(reasons) == 0:
        reasons.append("Stable socio-economic indicators")

    st.subheader(f"Prediction: {category}")
    st.write("Reasons:", reasons)
