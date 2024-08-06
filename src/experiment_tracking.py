"""
To run this file you need to launch the MLflow server locally by running the following command in your terminal:

mlflow server --backend-store-uri sqlite:///backend.db
"""
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import pickle
from mlflow import MlflowClient

# Set the remote tracking URI
remote_tracking_uri = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(remote_tracking_uri)

print(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")

# Create new experiment
experiment_name = "Sklearn Models"
mlflow.set_experiment(experiment_name)

# Ensure the models directory exists
models_dir = os.path.join(os.getcwd(), "models")
os.makedirs(models_dir, exist_ok=True)

def train_and_log_model(model, model_name, X_train, X_test, y_train, y_test, dataset_path):
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.set_tag("model", model_name)
        
        # Log the full dataset as an artifact
        mlflow.log_artifact(dataset_path, artifact_path="data")
        
        # Log the training dataset
        mlflow.log_input(mlflow.data.from_pandas(X_train), context="training")
        
        # Log the test dataset
        mlflow.log_input(mlflow.data.from_pandas(X_test), context="test")

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Log parameters
        for param, value in model.get_params().items():
            mlflow.log_param(param, value)

        # Log metrics
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        mlflow.log_metric('mae', mae)
        mlflow.log_metric("r2", r2)

        # Log model with MLflow
        mlflow.sklearn.log_model(model, "model")

        # Save model with pickle in the models folder
        pickle_path = os.path.join(models_dir, f"{model_name}.pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Log the pickle file as an artifact
        mlflow.log_artifact(pickle_path, artifact_path="models")

        print(f"Run ID for {model_name}:", run.info.run_id)
        print(f"Model saved as {pickle_path}")
        
        return run.info.run_id

def main():
    # Path to the dataset
    dataset_path = os.path.abspath('../MLops-Project/data/hour.csv')
    
    # Check if the dataset file exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")

    # Load and prepare data
    df = pd.read_csv(dataset_path)
    features = ['season', 'holiday', 'workingday', 'weathersit', 'temp', 'atemp', 
                'hum', 'windspeed', 'hr', 'mnth', 'yr']
    X = df[features]
    y = df['cnt']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Log the datasets in a separate run
    with mlflow.start_run(run_name="Dataset Logging"):
        mlflow.log_artifact(dataset_path, artifact_path="data")
        mlflow.log_input(mlflow.data.from_pandas(X_train), context="training")
        mlflow.log_input(mlflow.data.from_pandas(X_test), context="test")
        print(f"Full dataset logged: {dataset_path}")
        print("Training and test datasets logged")

    # Train and log Linear Regression
    lr_run_id = train_and_log_model(
        LinearRegression(n_jobs=1), "LinearRegression", X_train, X_test, y_train, y_test, dataset_path
    )

    # Register and transition Linear Regression model to Production
    client = MlflowClient()
    lr_model_uri = f"runs:/{lr_run_id}/model"
    lr_registered_model = mlflow.register_model(lr_model_uri, "LinearRegression_registered")
    client.transition_model_version_stage(
        name=lr_registered_model.name,
        version=lr_registered_model.version,
        stage="Production"
    )

    # Train and log Decision Tree Regressor
    dt_run_id = train_and_log_model(
        DecisionTreeRegressor(), "DecisionTreeRegressor", X_train, X_test, y_train, y_test, dataset_path
    )

    # Register and transition Decision Tree Regressor model to Production
    dt_model_uri = f"runs:/{dt_run_id}/model"
    dt_registered_model = mlflow.register_model(dt_model_uri, "DecisionTreeRegressor_registered")
    client.transition_model_version_stage(
        name=dt_registered_model.name,
        version=dt_registered_model.version,
        stage="Production"
    )

    print("\nExperiment Tracking Completed")
    print(f"Linear Regression Run ID: {lr_run_id}")
    print(f"Decision Tree Regressor Run ID: {dt_run_id}")

if __name__ == "__main__":
    main()