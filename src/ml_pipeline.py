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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@task
def read_data():
    df = pd.read_csv("data/hour.csv")
    return df

@task
def preprocess_data(df):
    X = df[FEATURES]
    y = df["cnt"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

@task
def train_model(X_train, y_train):
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    return model

@task
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mae, r2

@task
def log_model(model, mae, r2):
    with mlflow.start_run():
        mlflow.log_param("model_type", "DecisionTreeRegressor")
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, "model")
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        pickle_path = os.path.join(models_dir, "DecisionTreeRegressor_model.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(model, f)
        mlflow.log_artifact(pickle_path)

@flow(log_prints=True)
def ml_pipeline():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("MLflow Prefect Integration")

    df = read_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = train_model(X_train, y_train)
    mae, r2 = evaluate_model(model, X_test, y_test)
    log_model(model, mae, r2)

if __name__ == "__main__":
    ml_pipeline()
