"""
model.py
This module defines the ModelService class for bike-sharing prediction.
"""

import base64
import json


class ModelService:
    """
    Class to handle model predictions and data preprocessing.
    """

    def __init__(self, model, version=None):
        """
        Initializes the ModelService with a given model and optional version.

        Args:
            model: The machine learning model to use for predictions.
            version: The version of the model.
        """
        self.model = model
        self.version = version

    def base64_decode(self, encoded_data):
        """
        Decodes base64 encoded data and converts it to a JSON object.

        Args:
            encoded_data: The base64 encoded data.

        Returns:
            A JSON object parsed from the decoded string.
        """
        decoded_bytes = base64.b64decode(encoded_data)
        decoded_str = decoded_bytes.decode("utf-8")
        return json.loads(decoded_str)

    def prepare_features(self, ride):
        """
        Prepares feature dictionary from the ride data.

        Args:
            ride: A dictionary containing ride data.

        Returns:
            A dictionary of features.
        """
        features = {
            "season": ride["season"],
            "holiday": ride["holiday"],
            "workingday": ride["workingday"],
            "weathersit": ride["weathersit"],
            "temp": ride["temp"],
            "atemp": ride["atemp"],
            "hum": ride["hum"],
            "windspeed": ride["windspeed"],
            "hr": ride["hr"],
            "mnth": ride["mnth"],
            "yr": ride["yr"],
        }
        return features

    def predict(self, features):
        """
        Predicts the count using the model and features provided.

        Args:
            features: A dictionary of input features.

        Returns:
            The predicted count as a float.
        """
        features_list = [list(features.values())]
        preds = self.model.predict(features_list)
        return float(preds[0])

    def lambda_handler(self, event):
        """
        Handles Lambda events and processes predictions.

        Args:
            event: The event containing Kinesis data.

        Returns:
            A dictionary containing predictions for each record.
        """
        predictions = []
        for record in event["Records"]:
            base64_input = record["kinesis"]["data"]
            ride_event = self.base64_decode(base64_input)
            ride = ride_event["ride"]

            features = self.prepare_features(ride)
            prediction_result = self.predict(features)

            result = {
                "model": "bike_sharing_prediction_model",
                "version": self.version,
                "prediction": {"prediction_result": prediction_result},
            }
            predictions.append(result)

        return {"predictions": predictions}
