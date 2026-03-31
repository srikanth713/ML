
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and scaler
try:
    model = joblib.load('logistic_regression_model.pkl')
    scaler = joblib.load('standard_scaler.pkl')
    print("Model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

@app.route('/')
def home():
    return "Logistic Regression API is running! Send POST requests to /predict."

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model or scaler not loaded'}), 500

    try:
        data = request.get_json(force=True)

        # Ensure input data has the correct feature names and order
        # Based on your training data, the features are:
        # Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

        # Convert input dictionary to a pandas DataFrame
        # It's important that the order of columns matches the training data
        input_df = pd.DataFrame([data], columns=feature_names)

        # Scale the input features
        scaled_input = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(scaled_input)
        prediction_proba = model.predict_proba(scaled_input)

        # Return prediction as JSON
        return jsonify({
            'prediction': int(prediction[0]),
            'probability_no_diabetes': prediction_proba[0][0],
            'probability_diabetes': prediction_proba[0][1]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # In a production environment, you might use a more robust web server like Gunicorn or uWSGI
    # For local development, debug=True provides useful error messages and auto-reloading
    app.run(host='0.0.0.0', port=5000, debug=True)
