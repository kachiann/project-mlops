"""
Integration test for Docker prediction API.
"""

import json
import sys
import time

import requests
from deepdiff import DeepDiff


def load_event_data(file_path):
    """
    Load event data from a JSON file.

    :param file_path: Path to the JSON file.
    :return: The event data as a dictionary.
    """
    try:
        with open(file_path, "rt", encoding="utf-8") as f_in:
            event = json.load(f_in)
        print("Loaded event data successfully.")
        return event
    except FileNotFoundError:
        print("Error: event.json file not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print("Error: event.json file is not valid JSON.")
        sys.exit(1)


def send_post_request(url, event, retries=3, delay=2):
    """
    Send a POST request to the API endpoint with retries.

    :param url: The API endpoint URL.
    :param event: The event data to send.
    :param retries: Number of retry attempts.
    :param delay: Delay between retries in seconds.
    :return: The JSON response from the API.
    """
    for attempt in range(retries):
        try:
            response = requests.post(url, json=event, timeout=10)
            response.raise_for_status()
            print("Request successful.")
            return response.json()
        except requests.exceptions.RequestException as request_error:
            print(f"Error while sending request: {request_error}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached. Exiting.")
                sys.exit(1)


def test_prediction():
    """
    Tests the prediction API endpoint.
    """
    # Load event data from the JSON file
    event = load_event_data("tests/integration-test/event.json")

    # Define the API endpoint URL
    url = "http://localhost:8080/predict"

    # Send a POST request to the endpoint
    actual_response = send_post_request(url, event)

    # Print the actual response for debugging
    print("Actual response:")
    print(json.dumps(actual_response, indent=2))

    # Define the expected response format
    expected_response = {"prediction": [147.0]}

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


# Run the test
if __name__ == "__main__":
    test_prediction()
