- **Created a test function** for the API prediction endpoint in `test_docker.py`.
- **Checked the response** against the expected output using `DeepDiff`.
- **Passed the test**, confirming that the actual response matched your expectations.

The `test_docker.py` file contains integration test for the Dockerized model prediction API. Below are instructions on how to run the tests and understand the setup.

Before running the test, ensure you have met the following requirements:

- **Python**: Make sure Python is installed (version 3.11 or higher).
- **Pipenv**: Ensure Pipenv is installed for managing dependencies. Install it with:
  ```bash
  pip install pipenv
  ```
- **DeepDiff**: The DeepDiff library is required for comparing JSON responses. This should be included in the `Pipfile`.


## Setting Up the Environment

1. Navigate to the project or tests directory

2. Install the required dependencies:

   ```bash
   pipenv install --dev
   ```

3. Activate the virtual environment:

   ```bash
   pipenv shell
   ```
## Running the Test

To execute the tests in `test_docker.py`, use the following command:

```bash
pytest integration-test/test_docker.py
```
### Test Overview

- The test script loads an event from `event.json`, which should be present in the same directory as `test_docker.py`.
- It sends a POST request to the prediction API running locally at `http://localhost:8080/predict`.
- The test compares the actual response from the API against the expected response format.
- Any differences will be printed out using the DeepDiff library.

This test helps ensure that your Dockerized API behaves as expected when processing input data. Make adjustments as necessary to the test based on the evolving requirements of your application.


