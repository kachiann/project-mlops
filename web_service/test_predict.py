import requests

url = "http://localhost:8080/predict"
data = {
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
    "yr": 1,
}

try:
    response = requests.post(url, json=data)
    response.raise_for_status()  # Raise an HTTPError if the HTTP request returned an unsuccessful status code
    print(response.json())
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
