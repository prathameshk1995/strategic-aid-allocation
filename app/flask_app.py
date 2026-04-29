from flask import Flask
import joblib
from flask import request, jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)

def load_artifacts():
    model = joblib.load('../model/kmeans_model.pkl')
    scaler = joblib.load('../model/scaler.pkl')
    features = joblib.load('../model/features.pkl')
    thresholds = joblib.load('../model/thresholds.pkl')
    return model, scaler, features, thresholds

model, scaler, features, thresholds = load_artifacts()

# 🔥 Step 1: Log Transform
def log_transform(df):
    cols_to_log = ['exports', 'income', 'imports', 'gdpp']
    for col in cols_to_log:
        df[col] = np.log1p(df[col])
    return df

# 🔥 Step 2: IQR Clipping (inflation)
def clip_inflation(df, thresholds):
    lower = thresholds['inflation_lower']
    upper = thresholds['inflation_upper']
    df['inflation'] = df['inflation'].clip(lower, upper)
    return df

# 🔥 Step 3: Creating features
def create_features(df, thresholds):
    df['high_child_mort'] = (df['child_mort'] > thresholds['child_mort_75']).astype(int)
    df['low_income'] = (df['income'] < thresholds['income_25']).astype(int)
    df['high_inflation'] = (df['inflation'] > thresholds['inflation_75']).astype(int)
    df['low_life_expec'] = (df['life_expec'] < thresholds['life_expec_25']).astype(int)
    return df


#creating home API endpoint
@app.route('/home')
def home():
    return "Hello Prathamesh, your API is running"


#creating predict API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = data['features']

        print("\n🔹 Raw JSON Input:", data)
        print("🔹 Input Feature List:", input_data)

        # Validate input length
        if len(input_data) != 9:
            return jsonify({"error": "Expected 9 input features"}), 400
        
        # raw original features
        raw_features = ['child_mort','exports','health','imports','income','inflation','life_expec','total_fer','gdpp']

        # Convert to DataFrame
        df_input = pd.DataFrame([input_data], columns=raw_features)

        print("\n🔹 DataFrame BEFORE feature engineering:")
        print(df_input)

        # ✅ Step 1: Log Transform
        df_input = log_transform(df_input)
        print("\n🔹 AFTER LOG:\n", df_input)

        # ✅ Step 2: Clip inflation
        df_input = clip_inflation(df_input, thresholds)
        print("\n🔹 AFTER CLIPPING:\n", df_input)

        # ✅ Step 3: Feature Engineering
        df_input = create_features(df_input, thresholds)
        print("\n🔹 AFTER FEATURE ENGINEERING:\n", df_input)

        # ✅ Step 4: Align columns
        df_input = df_input[features]

        # ✅ Step 5: Scale
        scaled_input = scaler.transform(df_input)

        # ✅ Step 6: Predict
        prediction = model.predict(scaled_input)[0]

        print("\n🔹 FINAL PREDICTION:", prediction)

        # Business logic
        cluster_map = {
            1: "🔴 High Risk - Needs Immediate Aid",
            2: "🟡 Medium Risk - Economic Instability",
            0: "🟢 Low Risk - Stable"
        }

        category = cluster_map.get(prediction, "Unknown")
       
        #reasons
        reasons = []
        if df_input['child_mort'][0] > thresholds['child_mort_75']:
            reasons.append("High child mortality")

        if df_input['income'][0] < thresholds['income_25']:
            reasons.append("Low income")

        if df_input['life_expec'][0] < thresholds['life_expec_25']:
            reasons.append("Low life expectancy")

        if df_input['inflation'][0] > thresholds['inflation_75']: 
            reasons.append("High inflation (economic instability)") 
        
        #Default case 
        if len(reasons) == 0: 
            reasons.append("Stable socio-economic indicators")

        # Response
        return jsonify({
            "cluster": int(prediction),
            "category": category,
            "reasons": reasons
            })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

#Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
