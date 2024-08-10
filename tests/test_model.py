"""
test_model.py
This module contains tests for the ModelService class.
"""

from pathlib import Path
import model


def read_text(file):
    """
    Reads the content of a text file.

    Args:
        file: The name of the file to read.

    Returns:
        The content of the file as a string.
    """
    test_directory = Path(__file__).parent

    with open(test_directory / file, 'rt', encoding='utf-8') as f_in:
        return f_in.read().strip()


def test_base64_decode():
    """
    Tests the base64_decode method of the ModelService class.
    """
    base64_input = read_text('bike_data.b64')

    actual_result = model.ModelService(None).base64_decode(base64_input)
    expected_result = {
        "ride": {
            "season": 1,
            "holiday": 0,
            "workingday": 1,
            "weathersit": 1,
            "temp": 0.3,
            "atemp": 0.3,
            "hum": 0.5,
            "windspeed": 0.2,
            "hr": 10,
            "mnth": 6,
            "yr": 1
        },
        "ride_id": 256,
    }

    assert actual_result == expected_result


def test_prepare_features():
    """
    Tests the prepare_features method of the ModelService class.
    """
    model_service = model.ModelService(None)

    ride = {
        "season": 1,
        "holiday": 0,
        "workingday": 1,
        "weathersit": 1,
        "temp": 0.3,
        "atemp": 0.3,
        "hum": 0.5,
        "windspeed": 0.2,
        "hr": 10,
        "mnth": 6,
        "yr": 1
    }

    actual_features = model_service.prepare_features(ride)

    expected_features = {
        "season": 1,
        "holiday": 0,
        "workingday": 1,
        "weathersit": 1,
        "temp": 0.3,
        "atemp": 0.3,
        "hum": 0.5,
        "windspeed": 0.2,
        "hr": 10,
        "mnth": 6,
        "yr": 1
    }

    assert actual_features == expected_features


class ModelMock:
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

    features = {
        "season": 1,
        "holiday": 0,
        "workingday": 1,
        "weathersit": 1,
        "temp": 0.3,
        "atemp": 0.3,
        "hum": 0.5,
        "windspeed": 0.2,
        "hr": 10,
        "mnth": 6,
        "yr": 1
    }

    actual_prediction = model_service.predict(features)
    expected_prediction = 100.0

    assert actual_prediction == expected_prediction


def test_lambda_handler():
    """
    Tests the lambda_handler method of the ModelService class.
    """
    model_mock = ModelMock(100.0)
    model_version = 'Test123'
    model_service = model.ModelService(model_mock, model_version)

    base64_input = read_text('bike_data.b64')

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
        'predictions': [
            {
                'model': 'bike_sharing_prediction_model',
                'version': model_version,
                'prediction': {
                    'prediction_result': 100.0,
                    'ride_id': 256,
                },
            }
        ]
    }

    assert actual_predictions == expected_predictions