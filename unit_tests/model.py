import base64
import json


class ModelService:
    def __init__(self, model, version=None):
        self.model = model
        self.version = version

    def base64_decode(self, encoded_data):
        decoded_bytes = base64.b64decode(encoded_data)
        decoded_str = decoded_bytes.decode('utf-8')
        return json.loads(decoded_str)

    def prepare_features(self, ride):
        features = {
            'season': ride['season'],
            'holiday': ride['holiday'],
            'workingday': ride['workingday'],
            'weathersit': ride['weathersit'],
            'temp': ride['temp'],
            'atemp': ride['atemp'],
            'hum': ride['hum'],
            'windspeed': ride['windspeed'],
            'hr': ride['hr'],
            'mnth': ride['mnth'],
            'yr': ride['yr']
        }
        return features

    def predict(self, features):
        X = [list(features.values())]
        preds = self.model.predict(X)
        return float(preds[0])

    def lambda_handler(self, event):
        predictions = []
        for record in event['Records']:
            base64_input = record['kinesis']['data']
            ride_event = self.base64_decode(base64_input)
            ride = ride_event['ride']
            ride_id = ride_event['ride_id']

            features = self.prepare_features(ride)
            prediction_result = self.predict(features)

            result = {
                'model': 'bike_sharing_prediction_model',
                'version': self.version,
                'prediction': {
                    'prediction_result': prediction_result,
                    'ride_id': ride_id,
                },
            }
            predictions.append(result)

        return {'predictions': predictions}