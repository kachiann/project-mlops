"""
Integration test for Docker prediction API.
"""

import json

import requests
from deepdiff import DeepDiff


def test_prediction():
    """
    Tests the prediction API endpoint.
    """
    # Load event data from the JSON file
    try:
        with open(
            "../tests/integration-test/event.json", "rt", encoding="utf-8"
        ) as f_in:
            event = json.load(f_in)
    except FileNotFoundError:
        print("Error: event.json file not found.")
        exit(1)
    except json.JSONDecodeError:
        print("Error: event.json file is not valid JSON.")
        exit(1)

    # Define the API endpoint URL
    url = "http://localhost:8080/predict"

    # Send a POST request to the endpoint
    try:
        response = requests.post(url, json=event)
        response.raise_for_status()  # Raise an error for bad responses
        actual_response = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error while sending request: {e}")
        exit(1)

    # Print the actual response for debugging
    print("Actual response:")
    print(json.dumps(actual_response, indent=2))

    # Define the expected response format
    expected_response = {"prediction": [132.0]}

    # Compare actual and expected responses
    diff = DeepDiff(actual_response, expected_response, significant_digits=1)
    print(f"diff={diff}")

    # Assert that there are no differences in the expected values
    if "type_changes" in diff or "values_changed" in diff:
        print("Differences found between actual and expected responses:")
        print(diff)
        assert False  # Mark the test as failed
    else:
        print("The actual response matches the expected response.")
