import datetime
import logging
import random
import time

import joblib
import pandas as pd
import psycopg
from evidently import ColumnMapping
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
)
from evidently.report import Report
from prefect import flow, task

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)

SEND_TIMEOUT = 10
rand = random.Random()

create_table_statement = """
CREATE TABLE IF NOT EXISTS dummy_metrics (
    timestamp TIMESTAMP,
    prediction_drift FLOAT,
    num_drifted_columns INTEGER,
    share_missing_values FLOAT
)
"""

# Update the file path to the new location
reference_data = pd.read_csv("../project-mlops/data/reference.csv")
with open("../project-mlops/models/DecisionTreeRegressor.pkl", "rb") as f_in:
    model = joblib.load(f_in)

raw_data = pd.read_csv("../project-mlops/data/hour.csv")

features = [
    "season",
    "holiday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "hr",
    "mnth",
    "yr",
]
column_mapping = ColumnMapping(
    prediction="prediction", numerical_features=features, target=None
)

report = Report(
    metrics=[
        ColumnDriftMetric(column_name="prediction"),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
    ]
)


@task
def prep_db():
    try:
        with psycopg.connect(
            "host=localhost port=5432 user=postgres password=example", autocommit=True
        ) as conn:
            # Check if the database exists
            res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
            if not res.fetchall():
                logging.info("Database 'test' not found. Creating database.")
                conn.execute("CREATE DATABASE test;")
            else:
                logging.info("Database 'test' already exists.")

            # Connect to the database and create the table
            with psycopg.connect(
                "host=localhost port=5432 dbname=test user=postgres password=example"
            ) as conn:
                conn.execute(create_table_statement)
                logging.info("Table 'dummy_metrics' is ready.")
    except psycopg.OperationalError as e:
        logging.error("OperationalError: %s", str(e))
    except Exception as e:
        logging.error("Error preparing the database: %s", str(e))


@task
def calculate_metrics_postgresql(curr):
    try:
        current_data = raw_data.copy()
        current_data["prediction"] = model.predict(current_data[features])

        report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=column_mapping,
        )

        result = report.as_dict()

        prediction_drift = result["metrics"][0]["result"]["drift_score"]
        num_drifted_columns = result["metrics"][1]["result"][
            "number_of_drifted_columns"
        ]
        share_missing_values = result["metrics"][2]["result"]["current"][
            "share_of_missing_values"
        ]

        curr.execute(
            "INSERT INTO dummy_metrics(timestamp, prediction_drift, num_drifted_columns, share_missing_values) VALUES (%s, %s, %s, %s)",
            (
                datetime.datetime.now(),
                prediction_drift,
                num_drifted_columns,
                share_missing_values,
            ),
        )
        logging.info("Metrics inserted into database.")
    except Exception as e:
        logging.error("Error calculating metrics: %s", str(e))


@flow
def batch_monitoring_backfill():
    prep_db()
    last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
    try:
        with psycopg.connect(
            "host=localhost port=5432 dbname=test user=postgres password=example",
            autocommit=True,
        ) as conn:
            for _ in range(27):
                with conn.cursor() as curr:
                    calculate_metrics_postgresql(curr)

                new_send = datetime.datetime.now()
                seconds_elapsed = (new_send - last_send).total_seconds()
                if seconds_elapsed < SEND_TIMEOUT:
                    time.sleep(SEND_TIMEOUT - seconds_elapsed)
                last_send += datetime.timedelta(seconds=10)
                logging.info("Data sent. Waiting for the next iteration.")
    except Exception as e:
        logging.error("Error in batch monitoring: %s", str(e))


if __name__ == "__main__":
    batch_monitoring_backfill()
