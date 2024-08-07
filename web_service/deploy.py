"""
To run this file you need to launch the MLflow server locally by running the following command in your terminal:

mlflow server --backend-store-uri sqlite:///backend.db
"""

import os
import time
import requests
import mlflow
from mlflow.sklearn import load_model
from flask import Flask, request, jsonify
import pandas as pd

def wait_for_mlflow_server(url, max_retries=30, delay=10):
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{url}/health")
            if response.status_code == 200:
                print("MLflow server is up and running!")
                return True
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1}/{max_retries}: MLflow server not ready. Error: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached. MLflow server is not available.")
                return False

# Set the tracking URI from environment variable
mlflow_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000')
mlflow.set_tracking_uri(mlflow_uri)

print(f"Waiting for MLflow server at {mlflow_uri}")
if not wait_for_mlflow_server(mlflow_uri):
    raise Exception("MLflow server is not available. Exiting.")

try:
    # Load the model from MLflow
    model_name = "DecisionTreeRegressor_registered"
    client = mlflow.tracking.MlflowClient()
    
    # Get the Production version of the model
    production_model = client.get_latest_versions(model_name, stages=["Production"])[0]
    
    print(f"Loading model '{model_name}' version {production_model.version}")
    model = mlflow.sklearn.load_model(f"models:/{model_name}/{production_model.version}")
    print(f"Successfully loaded model {model_name} version {production_model.version}")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Create a Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return "Welcome to the ML Prediction API. Use the /predict endpoint to make predictions."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = ['season', 'holiday', 'workingday', 'weathersit', 'temp', 'atemp', 
                'hum', 'windspeed', 'hr', 'mnth', 'yr']
    
    # Create a list with a single dictionary to ensure a DataFrame with one row
    input_data = pd.DataFrame([data])
    
    # Ensure all required features are present
    for feature in features:
        if feature not in input_data.columns:
            return jsonify({'error': f'Missing feature: {feature}'}), 400
    
    # Select only the required features in the correct order
    input_data = input_data[features]
    
    prediction = model.predict(input_data)
    return jsonify({'prediction': prediction.tolist()})

@app.route('/favicon.ico')
def favicon():
    return '', 204  # No content response

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
