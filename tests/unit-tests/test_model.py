"""
test_model.py
This module contains tests for the ModelService class.
"""

from pathlib import Path
import model
from utils import BIKE_DATA_TEMPLATE

# Consider using a constant for file paths if reused
TEST_DIRECTORY = Path(__file__).parent


def read_text(file):
    """
    Reads the content of a text file.

    Args:
        file: The name of the file to read.

    Returns:
        The content of the file as a string.
    """
    with open(TEST_DIRECTORY / file, "rt", encoding="utf-8") as f_in:
        return f_in.read().strip()


def test_base64_decode():
    """
    Tests the base64_decode method of the ModelService class.
    """
    base64_input = read_text("bike_data.b64")

    actual_result = model.ModelService(None).base64_decode(base64_input)

    # The expected result should not include 'ride_id'
    expected_result = {
        "ride": BIKE_DATA_TEMPLATE,  # Ensure this matches the output structure
    }

    assert actual_result == expected_result


def test_prepare_features():
    """
    Tests the prepare_features method of the ModelService class.
    """
    model_service = model.ModelService(None)

    # Use the existing structure directly
    actual_features = model_service.prepare_features(
        BIKE_DATA_TEMPLATE
    )  # Pass the whole template or adjust accordingly

    expected_features = BIKE_DATA_TEMPLATE  # Use the same structure for consistency

    assert actual_features == expected_features


class ModelMock:
    # pylint: disable=too-few-public-methods
    """
    A mock model for testing predictions.
    """

    def __init__(self, value):
        self.value = value

    def predict(self, features):
        """
        Predicts a constant value based on input features.

        Args:
            features: The input features.

        Returns:
            A list of predicted values.
        """
        n = len(features)
        return [self.value] * n


def test_predict():
    """
    Tests the predict method of the ModelService class.
    """
    model_mock = ModelMock(100.0)
    model_service = model.ModelService(model_mock)

    features = BIKE_DATA_TEMPLATE  # Use the template for consistency

    actual_prediction = model_service.predict(features)
    expected_prediction = 100.0

    assert actual_prediction == expected_prediction


def test_lambda_handler():
    """
    Tests the lambda_handler method of the ModelService class.
    """
    model_mock = ModelMock(100.0)
    model_version = "Test123"
    model_service = model.ModelService(model_mock, model_version)

    base64_input = read_text("bike_data.b64")

    event = {
        "Records": [
            {
                "kinesis": {
                    "data": base64_input,
                },
            }
        ]
    }

    actual_predictions = model_service.lambda_handler(event)
    expected_predictions = {
        "predictions": [
            {
                "model": "bike_sharing_prediction_model",
                "version": model_version,
                "prediction": {"prediction_result": 100.0},
            }
        ]
    }

    assert actual_predictions == expected_predictions
