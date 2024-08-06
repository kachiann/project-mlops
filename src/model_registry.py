import mlflow
from mlflow import MlflowClient
import pickle
import os

# Set the remote tracking URI
remote_tracking_uri = "http://127.0.0.1:5000" 
mlflow.set_tracking_uri(remote_tracking_uri)

print(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")

# Ensure the models directory exists
models_dir = os.path.join(os.getcwd(), "models")
os.makedirs(models_dir, exist_ok=True)

def register_model(run_id, model_name):
    model_uri = f"runs:/{run_id}/model"
    registered_model_name = f"{model_name}_registered"
    registered_model = mlflow.register_model(model_uri, registered_model_name)
    
    print(f"Registered model name: {registered_model_name}")
    print(f"Registered model version: {registered_model.version}")
    
    # Transition the model to Production stage
    client = MlflowClient()
    client.transition_model_version_stage(
        name=registered_model_name,
        version=registered_model.version,
        stage="Production"
    )
    print(f"Transitioned {registered_model_name} version {registered_model.version} to Production stage")
    
    return registered_model_name, registered_model.version

def compare_models(lr_run_id, dt_run_id):
    client = MlflowClient()
    lr_run = client.get_run(lr_run_id)
    dt_run = client.get_run(dt_run_id)

    print("\nModel Comparison:")
    print("Linear Regression - MAE:", lr_run.data.metrics['mae'], "R2:", lr_run.data.metrics['r2'])
    print("Decision Tree Regressor - MAE:", dt_run.data.metrics['mae'], "R2:", dt_run.data.metrics['r2'])

def load_model_from_pickle(run_id, model_name):
    client = MlflowClient()
    artifact_path = client.download_artifacts(run_id, f"models/{model_name}.pkl")
    
    with open(artifact_path, 'rb') as f:
        model = pickle.load(f)
    
    # Save the model in the local models directory
    local_model_path = os.path.join(models_dir, f"{model_name}.pkl")
    with open(local_model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Loaded {model_name} from MLflow and saved to {local_model_path}")
    return model

def main():
    # Get run IDs from the user
    lr_run_id = input("Enter the run ID for Linear Regression: ")
    dt_run_id = input("Enter the run ID for Decision Tree Regressor: ")

    # Register models
    lr_model_name, lr_model_version = register_model(lr_run_id, "LinearRegression")
    dt_model_name, dt_model_version = register_model(dt_run_id, "DecisionTreeRegressor")

    # Compare models
    compare_models(lr_run_id, dt_run_id)

    # Print registered model info
    print("\nRegistered Models:")
    print(f"Linear Regression: {lr_model_name} (version {lr_model_version})")
    print(f"Decision Tree Regressor: {dt_model_name} (version {dt_model_version})")

    # Load models from pickle files
    lr_model = load_model_from_pickle(lr_run_id, "LinearRegression")
    dt_model = load_model_from_pickle(dt_run_id, "DecisionTreeRegressor")

if __name__ == "__main__":
    main()
