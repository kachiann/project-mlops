import requests

from tests.utils import BIKE_DATA_TEMPLATE

url = "http://localhost:8080/predict"


try:
    response = requests.post(url, json=BIKE_DATA_TEMPLATE)
    response.raise_for_status()  # Raise an HTTPError if the HTTP request returned an unsuccessful status code
    print(response.json())
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
