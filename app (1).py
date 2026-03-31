from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model & scaler
try:
    model = joblib.load('logistic_regression_model.pkl')
    scaler = joblib.load('standard_scaler.pkl')
    print("✅ Model and scaler loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model or scaler: {e}")
    model = None
    scaler = None

# Feature list (IMPORTANT)
FEATURE_NAMES = [
    'Pregnancies', 'Glucose', 'BloodPressure',
    'SkinThickness', 'Insulin', 'BMI',
    'DiabetesPedigreeFunction', 'Age'
]

@app.route('/')
def home():
    return "✅ API is running. Use POST /predict"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded properly'}), 500

    try:
        data = request.get_json()

        # 🔒 Validate input
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        missing = [f for f in FEATURE_NAMES if f not in data]
        if missing:
            return jsonify({'error': f'Missing fields: {missing}'}), 400

        # Convert to DataFrame safely
        input_data = {k: float(data[k]) for k in FEATURE_NAMES}
        input_df = pd.DataFrame([input_data])

        # Scale
        scaled_input = scaler.transform(input_df)

        # Predict
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0]

        return jsonify({
            'prediction': int(prediction),
            'probability_no_diabetes': float(probability[0]),
            'probability_diabetes': float(probability[1])
        })

    except ValueError:
        return jsonify({'error': 'Invalid data type. All inputs must be numbers'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
