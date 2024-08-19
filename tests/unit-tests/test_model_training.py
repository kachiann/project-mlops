import os
import unittest
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from mlflow.tracking import MlflowClient
from src.experiment_tracking import train_and_log_model, main

# Constants
FEATURES = [
    "season",
    "holiday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "hr",
    "mnth",
    "yr",
]

class TestModelTraining(unittest.TestCase):

    def setUp(self):
        # Create a small sample dataset for testing
        self.X = pd.DataFrame({
            'season': [1, 2, 3, 4],
            'holiday': [0, 0, 1, 0],
            'workingday': [1, 1, 0, 0],
            'weathersit': [1, 2, 3, 4],
            'temp': [0.24, 0.75, 0.50, 0.30],
            'atemp': [0.2879, 0.8, 0.55, 0.35],
            'hum': [0.81, 0.5, 0.7, 0.9],
            'windspeed': [0.0, 0.1, 0.2, 0.3],
            'hr': [0, 12, 23, 6],
            'mnth': [1, 6, 12, 7],
            'yr': [0, 1, 0, 1]
        })
        self.y = pd.Series([10, 50, 30, 20])  # Target variable
        self.dummy_dataset_path = "/tmp/dummy_dataset.csv"  # Dummy path for testing

        # Save dummy dataset
        pd.concat([self.X, self.y.rename('cnt')], axis=1).to_csv(self.dummy_dataset_path, index=False)

        # Ensure the models directory exists
        self.models_dir = os.path.join(os.getcwd(), "models")
        os.makedirs(self.models_dir, exist_ok=True)

    def tearDown(self):
        # Clean up any created files or directories after each test
        if os.path.exists(self.models_dir):
            for file in os.listdir(self.models_dir):
                file_path = os.path.join(self.models_dir, file)
                os.remove(file_path)
            os.rmdir(self.models_dir)
        # Remove the dummy dataset
        if os.path.exists(self.dummy_dataset_path):
            os.remove(self.dummy_dataset_path)

    def test_train_and_log_linear_regression(self):
        # Test Linear Regression training and logging using the sample dataset
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.25, random_state=42)
        lr_model = LinearRegression(n_jobs=1)
        lr_run_id = train_and_log_model(lr_model, "UnitTestLinearRegression", X_train, X_test, y_train, y_test, self.dummy_dataset_path)

        # Verify if run_id is returned and exists
        self.assertIsNotNone(lr_run_id)
        self.assertIsInstance(lr_run_id, str)

    def test_train_and_log_decision_tree_regressor(self):
        # Test Decision Tree Regressor training and logging using the sample dataset
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.25, random_state=42)
        dt_model = DecisionTreeRegressor()
        dt_run_id = train_and_log_model(dt_model, "UnitTestDecisionTreeRegressor", X_train, X_test, y_train, y_test, self.dummy_dataset_path)

        # Verify if run_id is returned and exists
        self.assertIsNotNone(dt_run_id)
        self.assertIsInstance(dt_run_id, str)

    def test_main_function(self):
        # Test the main function of experiment_tracking.py
        try:
            main()  # Simply check if it runs without errors
        except Exception as e:
            self.fail(f"main() raised an exception: {e}")

    def test_model_registration(self):
        # Test model registration and transitioning to production
        client = MlflowClient()

        # Verify if Linear Regression model has been registered and transitioned to Production
        lr_registered_model = client.get_registered_model("LinearRegression_registered")
        self.assertIsNotNone(lr_registered_model)

        # Verify the model version exists and is in the Production stage
        lr_versions = client.search_model_versions(f"name='{lr_registered_model.name}'")
        self.assertTrue(any(version.current_stage == "Production" for version in lr_versions))

        # Verify if Decision Tree Regressor model has been registered and transitioned to Production
        dt_registered_model = client.get_registered_model("DecisionTreeRegressor_registered")
        self.assertIsNotNone(dt_registered_model)

        # Verify the model version exists and is in the Production stage
        dt_versions = client.search_model_versions(f"name='{dt_registered_model.name}'")
        self.assertTrue(any(version.current_stage == "Production" for version in dt_versions))

if __name__ == '__main__':
    unittest.main()
