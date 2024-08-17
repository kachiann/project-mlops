.PHONY: setup train deploy monitor lint register mlflow

setup:
	pip install -r requirements.txt

mlflow:
	mlflow server --backend-store-uri sqlite:///backend.db


train:
	python src/experiment_tracking.py

deploy:
	python web_service/deploy.py

predict:
	python web_service/test_predict.py

workflow:
	python src/ml_pipeline.py

monitor:
	python monitoring/evidently_metrics_calculations.py

lint:
	pylint *.py

register:
	python src/model_registry.py


all: setup train register deploy predict lint