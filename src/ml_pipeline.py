import os
import pickle
import sys  # Standard library imports

# Third-party imports
import mlflow
import pandas as pd
from prefect import flow, task
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from constants import FEATURES

# Local application imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Constants
DATA_PATH = "data/hour.csv"
MODEL_DIR = "models"
MODEL_FILENAME = "DecisionTreeRegressor_model.pkl"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT = "MLflow Prefect Integration"


@task
def read_data(data_path=DATA_PATH):
    try:
        df = pd.read_csv(data_path)
        print(f"Data read successfully. Shape of the dataset: {df.shape}")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to read data: {e}") from e


@task
def preprocess_data(df):
    try:
        X = df[FEATURES]
        y = df["cnt"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print("Data preprocessing completed.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        raise RuntimeError(f"Data preprocessing failed: {e}") from e


@task
def train_model(X_train, y_train):
    try:
        model = DecisionTreeRegressor()
        model.fit(X_train, y_train)
        print("Model training completed.")
        return model
    except Exception as e:
        raise RuntimeError(f"Model training failed: {e}") from e


@task
def evaluate_model(model, X_test, y_test):
    try:
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        print(f"Model evaluation completed. MAE: {mae}, R²: {r2}")
        return mae, r2
    except Exception as e:
        raise RuntimeError(f"Model evaluation failed: {e}") from e


@task
def log_model(model, mae, r2, model_dir=MODEL_DIR, model_filename=MODEL_FILENAME):
    try:
        with mlflow.start_run():
            mlflow.log_param("model_type", "DecisionTreeRegressor")
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            mlflow.sklearn.log_model(model, "model")

            # Save model locally
            os.makedirs(model_dir, exist_ok=True)
            pickle_path = os.path.join(model_dir, model_filename)
            with open(pickle_path, "wb") as f:
                pickle.dump(model, f)

            # Log the model as an artifact in MLflow
            mlflow.log_artifact(pickle_path)
    except Exception as e:
        raise RuntimeError(f"Logging model failed: {e}") from e


@flow(log_prints=True)
def ml_pipeline():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    df = read_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = train_model(X_train, y_train)
    mae, r2 = evaluate_model(model, X_test, y_test)
    log_model(model, mae, r2)


if __name__ == "__main__":
    ml_pipeline()
