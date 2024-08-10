"""
To run this file you need to launch the MLflow server locally by 
running the following command in your terminal:

mlflow server --backend-store-uri sqlite:///backend.db
"""

import os
import time

import mlflow
import pandas as pd
import requests
from flask import Flask, jsonify, request
from src.constants import FEATURES


def wait_for_mlflow_server(url, max_retries=30, delay=10):
    # pylint: disable = inconsistent-return-statements
    """Wait until the MLflow server is available."""
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                print("MLflow server is up and running!")
                return True
        except requests.exceptions.RequestException as expectation:
            print(
                f"Attempt {attempt + 1}/{max_retries}: MLflow server not ready. Error: {expectation}"
            )
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached. MLflow server is not available.")
                return False


# Set the tracking URI from environment variable
mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(mlflow_uri)

print(f"Waiting for MLflow server at {mlflow_uri}")
if not wait_for_mlflow_server(mlflow_uri):
    raise Exception("MLflow server is not available. Exiting.")

try:
    # Load the model from MLflow
    model_name = "DecisionTreeRegressor_registered"
    client = mlflow.tracking.MlflowClient()

    # Get the Production version of the model
    production_model = client.get_latest_versions(model_name, stages=["Production"])

    if not production_model:
        raise Exception(f"No production model found for '{model_name}'.")

    print(f"Loading model '{model_name}' version {production_model[0].version}")
    model = mlflow.sklearn.load_model(
        f"models:/{model_name}/{production_model[0].version}"
    )
    print(
        f"Successfully loaded model {model_name} version {production_model[0].version}"
    )
except mlflow.exceptions.MlflowException as e:
    print(f"Error loading model: {e}")
    raise

# Create a Flask app
app = Flask(__name__)


@app.route("/")
def index():
    """Return a welcome message."""
    return "Welcome to the ML Prediction API. Use the /predict endpoint to make predictions."


@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction requests."""
    data = request.json

    # Check if input data is present and is a dictionary
    if not data or not isinstance(data, dict):
        return jsonify({"error": "Invalid input. Expected a JSON object."}), 400

    # Create a DataFrame from the input data
    input_data = pd.DataFrame([data])

    # Ensure all required features are present
    missing_features = [
        feature for feature in FEATURES if feature not in input_data.columns
    ]
    if missing_features:
        return (
            jsonify({"error": f'Missing features: {", ".join(missing_features)}'}),
            400,
        )

    # Select only the required features in the correct order
    input_data = input_data[FEATURES]

    try:
        prediction = model.predict(input_data)
        return jsonify({"prediction": prediction.tolist()})
    except Exception as expectation:
        return jsonify({"error": f"Prediction error: {str(expectation)}"}), 500


@app.route("/favicon.ico")
def favicon():
    """Return a no content response for favicon requests."""
    return "", 204  # No content response


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
