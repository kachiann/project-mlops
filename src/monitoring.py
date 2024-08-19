"""
This module handles model monitoring tasks for the bike-sharing demand prediction project.
It uses Evidently to generate data drift reports.
"""

import logging
import os
import warnings

import numpy as np
import pandas as pd
from evidently.metrics import DataDriftTable
from evidently.report import Report

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def preprocess_data(data):
    """Preprocess the data to handle zero values and create new features."""
    # Convert 'dteday' to datetime if it's not already
    if "dteday" in data.columns and data["dteday"].dtype == "object":
        data["dteday"] = pd.to_datetime(data["dteday"])

    # Create binary flags for zero rentals if they don't exist
    if "zero_casual" not in data.columns:
        data["zero_casual"] = (data["casual"] == 0).astype(int)
    if "zero_registered" not in data.columns:
        data["zero_registered"] = (data["registered"] == 0).astype(int)

    # Handle potential infinite values in numeric columns
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data[numeric_columns] = data[numeric_columns].replace([np.inf, -np.inf], np.nan)

    return data


def load_data(file_path):
    """Load data from CSV file and preprocess it."""
    try:
        data = pd.read_csv(file_path)
        data = preprocess_data(data)
        logger.info("Successfully loaded and preprocessed data from", file_path)
        return data
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        raise


def validate_data(data, name):
    """Validate the data and log information."""
    logger.info(f"Validating {name}:")
    logger.info(f"Shape: {data.shape}")
    logger.info(f"Columns: {data.columns.tolist()}")
    logger.info(f"Data types:\n{data.dtypes}")
    logger.info(f"First few rows:\n{data.head()}")

    numeric_columns = data.select_dtypes(include=[np.number]).columns
    if not numeric_columns.empty:
        logger.info(f"Numeric columns description:\n{data[numeric_columns].describe()}")
    else:
        logger.warning(f"{name} does not contain any numeric columns")


def analyze_data_quality(data, name):
    """Analyze the data quality and log potential issues."""
    logger.info(f"Analyzing data quality for {name}")

    # Columns where zeros are expected
    expected_zero_columns = [
        "holiday",
        "workingday",
        "casual",
        "registered",
        "zero_casual",
        "zero_registered",
    ]

    # Check for zero values in numeric columns where zeros are unexpected
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    unexpected_zero_columns = [
        col for col in numeric_columns if col not in expected_zero_columns
    ]
    zeros = (data[unexpected_zero_columns] == 0).sum()
    if zeros.any():
        logger.warning(
            f"Unexpected zero values in numeric columns:\n{zeros[zeros > 0]}"
        )

    # Check for missing values
    missing = data.isnull().sum()
    if missing.any():
        logger.warning(f"Missing values:\n{missing[missing > 0]}")

    # Basic statistical summary
    logger.info(f"Statistical summary:\n{data.describe()}")


def analyze_zero_proportions(data, name):
    """Analyze the proportion of zero values for relevant columns."""
    relevant_columns = ["holiday", "workingday", "casual", "registered"]
    for col in relevant_columns:
        zero_prop = (data[col] == 0).mean()
        logger.info(f"{name} - Proportion of zeros in {col}: {zero_prop:.2%}")


def analyze_seasonality(data, name):
    """Analyze the seasonality of the data."""
    season_counts = data["season"].value_counts().sort_index()
    logger.info(f"{name} - Season distribution:\n{season_counts}")


def compare_distributions(reference_data, production_data):
    """Compare key metrics between reference and production data."""
    key_metrics = ["casual", "registered", "cnt"]
    for metric in key_metrics:
        ref_mean = reference_data[metric].mean()
        prod_mean = production_data[metric].mean()
        change = (prod_mean - ref_mean) / ref_mean * 100
        logger.info(
            f"{metric} - Ref mean: {ref_mean:.2f}, Prod mean: {prod_mean:.2f}, Change: {change:.2f}%"
        )


def generate_data_drift_report(reference_data, production_data, output_path):
    """Generate a data drift report using Evidently."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            # Ensure both datasets have the same columns
            common_columns = list(
                set(reference_data.columns) & set(production_data.columns)
            )
            reference_data = reference_data[common_columns]
            production_data = production_data[common_columns]

            data_drift_report = Report(metrics=[DataDriftTable()])

            data_drift_report.run(
                reference_data=reference_data, current_data=production_data
            )
            data_drift_report.save_html(output_path)
            logger.info(f"Data drift report generated and saved to {output_path}")
    except Exception as e:
        logger.error(f"Error generating data drift report: {str(e)}")
        logger.error(
            f"Reference data shape: {reference_data.shape}, Production data shape: {production_data.shape}"
        )
        raise


if __name__ == "__main__":
    try:
        # Load your reference and production data
        reference_data = load_data("reference_data.csv")
        production_data = load_data("production_data.csv")

        # Validate the data
        validate_data(reference_data, "Reference data")
        validate_data(production_data, "Production data")

        # Analyze data quality
        analyze_data_quality(reference_data, "Reference data")
        analyze_data_quality(production_data, "Production data")

        # Analyze zero proportions
        analyze_zero_proportions(reference_data, "Reference data")
        analyze_zero_proportions(production_data, "Production data")

        # Analyze seasonality
        analyze_seasonality(reference_data, "Reference data")
        analyze_seasonality(production_data, "Production data")

        # Compare distributions
        compare_distributions(reference_data, production_data)

        # Check for column differences
        ref_cols = set(reference_data.columns)
        prod_cols = set(production_data.columns)
        if ref_cols != prod_cols:
            logger.warning("Column mismatch between reference and production data:")
            logger.warning(f"Columns only in reference data: {ref_cols - prod_cols}")
            logger.warning(f"Columns only in production data: {prod_cols - ref_cols}")

        # Ensure the output directory exists
        output_dir = "monitoring_reports"
        os.makedirs(output_dir, exist_ok=True)

        # Generate the report
        output_path = os.path.join(output_dir, "data_drift_report.html")
        generate_data_drift_report(reference_data, production_data, output_path)

        logger.info("Model monitoring completed successfully.")
    except Exception as e:
        logger.error(f"Model monitoring failed: {str(e)}")
