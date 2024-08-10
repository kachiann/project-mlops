# Define local image name and tag based on the current date and time
LOCAL_TAG := $(shell date +"%Y-%m-%d-%H-%M")
LOCAL_IMAGE_NAME := bike-sharing-prediction:$(LOCAL_TAG)

# Target to run tests
test:
	@echo "Running tests..."
	pytest tests

# Target to perform quality checks
quality_checks:
	@echo "Running quality checks..."
	isort .
	black .
	pylint **/*.py

# Target to build the Docker image after quality checks and tests
build: quality_checks test
	@echo "Building Docker image..."
	docker build -t $(LOCAL_IMAGE_NAME) .

# Target to run integration tests
integration_test: build
	@echo "Running integration tests..."
	LOCAL_IMAGE_NAME=$(LOCAL_IMAGE_NAME) bash integration-test/run.sh

# Target to publish the Docker image
publish: integration_test
	@echo "Publishing Docker image..."
	LOCAL_IMAGE_NAME=$(LOCAL_IMAGE_NAME) bash scripts/publish.sh

# Target to set up the development environment
setup:
	@echo "Setting up development environment..."
	pipenv install --dev
	pre-commit install

# Target to clean up Docker images and containers
clean:
	@echo "Cleaning up..."
	docker rmi -f $$(docker images -f "dangling=true" -q) || true
	docker container prune -f

.PHONY: test quality_checks build integration_test publish setup clean
