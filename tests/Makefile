# Define local image name and tag based on the current date and time
LOCAL_TAG := $(shell date +"%Y-%m-%d-%H-%M")
LOCAL_IMAGE_NAME := bike-sharing-prediction:${LOCAL_TAG}

# Target to run tests
test:
	pytest tests/

# Target to perform quality checks
quality_checks:
	isort .
	black .
	pylint --recursive=y .

# Target to build the Docker image after quality checks and tests
build: quality_checks test
	docker build -t ${LOCAL_IMAGE_NAME} .

# Target to run integration tests
integration_test: build
	LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME} bash integration-test/run.sh

# Target to publish the Docker image
publish: build integration_test
	LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME} bash scripts/publish.sh

# Target to set up the development environment
setup:
	pipenv install --dev
	pre-commit install
