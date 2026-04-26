from flask import Flask
import joblib
from flask import request, jsonify
import pandas as pd


app = Flask(__name__)

def load_artifacts():
    model = joblib.load('../model/kmeans_model.pkl')
    scaler = joblib.load('../model/scaler.pkl')
    features = joblib.load('../model/features.pkl')
    thresholds = joblib.load('../model/thresholds.pkl')
    return model, scaler, features, thresholds

model, scaler, features, thresholds = load_artifacts()

#creating home API endpoint
@app.route('/home')
def home():
    return "Hello Prathamesh, your API is running"


#creating predict API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Validate input
        if 'features' not in data:
            return jsonify({"error": "Missing features"}), 400

        input_data = data['features']

        # Convert to DataFrame
        df_input = pd.DataFrame([input_data], columns=features)

        # Scale input
        scaled_input = scaler.transform(df_input)

        # Predict
        prediction = model.predict(scaled_input)[0]

        # Business logic
        if prediction == 1:
            category = "High Risk - Needs Aid"
        elif prediction == 2:
            category = "Medium Risk"
        else:
            category = "Low Risk - No Immediate Aid Needed"

        #reasons
        reasons = []
        if df_input['child_mort'][0] > thresholds['child_mort']:
            reasons.append("High child mortality")

        if df_input['income'][0] < thresholds['income']:
            reasons.append("Low income")

        if df_input['life_expec'][0] < thresholds['life_expec']:
            reasons.append("Low life expectancy")

        if df_input['inflation'][0] > thresholds['inflation']: 
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
