
import os
import sys
import json
import numpy as np
import pandas as pd

from dataclasses import dataclass
from pandas.api.types import is_numeric_dtype, is_bool_dtype

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataValidationConfig:

    project_root: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )

    raw_data_path: str = os.path.join(
        project_root, "artifacts", "raw", "data.csv"
    )

    validation_report_path: str = os.path.join(
        project_root, "artifacts", "validation",
        "validation_report.json"
    )


class DataValidation:

    def __init__(self):
        self.validation_config = DataValidationConfig()

        self.expected_columns = [
            "product_id",
            "brand_name",
            "manufacturer",
            "price_inr",
            "is_discontinued",
            "dosage_form",
            "pack_size",
            "pack_unit",
            "num_active_ingredients",
            "primary_ingredient",
            "primary_strength",
            "active_ingredients",
            "therapeutic_class",
            "packaging_raw",
            "manufacturer_raw"
        ]

        self.optional_missing_columns = [
            "pack_size",
            "pack_unit",
            "primary_strength"
        ]

    def validate_dataset(self, df):

        logging.info("Starting PharmaPrice dataset validation")

        # 1. Dataset information
        total_rows, total_columns = df.shape

        # 2. Schema validation
        missing_columns = [
            col for col in self.expected_columns
            if col not in df.columns
        ]

        unexpected_columns = [
            col for col in df.columns
            if col not in self.expected_columns
        ]

        # 3. Missing value analysis
        missing_values = df.isnull().sum()

        missing_value_report = {
            col: int(count)
            for col, count in missing_values.items()
            if count > 0
        }

        missing_percentage = {
            col: round((count / total_rows) * 100, 2)
            for col, count in missing_values.items()
            if count > 0 and total_rows > 0
        }

        critical_columns = [
            "product_id",
            "brand_name",
            "manufacturer",
            "price_inr"
        ]

        critical_missing_values = {
            col: int(df[col].isnull().sum())
            for col in critical_columns
            if col in df.columns and df[col].isnull().any()
        }

        # 4. Duplicate analysis
        duplicate_rows = int(df.duplicated().sum())

        duplicate_product_ids = 0

        if "product_id" in df.columns:
            duplicate_product_ids = int(
                df["product_id"].duplicated().sum()
            )

        # 5. Price validation
        negative_prices = 0
        zero_prices = 0
        non_finite_prices = 0
        price_outliers = 0

        if "price_inr" in df.columns:

            if is_numeric_dtype(df["price_inr"]):

                prices = df["price_inr"].dropna()

                non_finite_prices = int(
                    (~np.isfinite(prices)).sum()
                )

                valid_prices = prices[np.isfinite(prices)]

                negative_prices = int(
                    (valid_prices < 0).sum()
                )

                zero_prices = int(
                    (valid_prices == 0).sum()
                )

                # IQR-based outlier analysis
                positive_prices = valid_prices[
                    valid_prices > 0
                ]

                if len(positive_prices) > 0:

                    q1 = positive_prices.quantile(0.25)
                    q3 = positive_prices.quantile(0.75)

                    iqr = q3 - q1

                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr

                    price_outliers = int(
                        (
                            (positive_prices < lower_bound) |
                            (positive_prices > upper_bound)
                        ).sum()
                    )

            else:
                negative_prices = 0

        # 6. Data type validation
        data_type_checks = {}

        numeric_columns = [
            "price_inr",
            "pack_size",
            "num_active_ingredients"
        ]

        for col in numeric_columns:

            if col in df.columns:

                data_type_checks[col] = bool(
                    is_numeric_dtype(df[col])
                )

        if "is_discontinued" in df.columns:

            data_type_checks["is_discontinued"] = bool(
                is_bool_dtype(df["is_discontinued"])
            )

        failed_data_types = [
            col for col, valid in data_type_checks.items()
            if not valid
        ]

        # 7. High cardinality analysis
        high_cardinality_columns = {}

        for col in df.select_dtypes(include=["object", "str"]).columns:

            unique_count = int(df[col].nunique())

            if unique_count > 100:

                high_cardinality_columns[col] = unique_count

        # 8. Validation errors
        validation_errors = []

        if df.empty:
            validation_errors.append("Dataset is empty")

        if missing_columns:
            validation_errors.append(
                f"Missing required columns: {missing_columns}"
            )

        if critical_missing_values:
            validation_errors.append(
                f"Missing critical values: {critical_missing_values}"
            )

        if negative_prices > 0:
            validation_errors.append(
                f"Found {negative_prices} negative prices"
            )

        if non_finite_prices > 0:
            validation_errors.append(
                f"Found {non_finite_prices} non-finite prices"
            )

        if failed_data_types:
            validation_errors.append(
                f"Invalid data types: {failed_data_types}"
            )

        # 9. Warnings
        validation_warnings = []

        if duplicate_rows > 0:
            validation_warnings.append(
                f"Found {duplicate_rows} duplicate rows"
            )

        if duplicate_product_ids > 0:
            validation_warnings.append(
                f"Found {duplicate_product_ids} duplicate product IDs"
            )

        if price_outliers > 0:
            validation_warnings.append(
                f"Found {price_outliers} price outliers using IQR"
            )

        if zero_prices > 0:
            validation_warnings.append(
                f"Found {zero_prices} zero-priced products"
            )

        if unexpected_columns:
            validation_warnings.append(
                f"Unexpected columns found: {unexpected_columns}"
            )

        # 10. Generate validation report
        validation_report = {

            "dataset_name": "PharmaPrice",

            "dataset_information": {
                "total_rows": int(total_rows),
                "total_columns": int(total_columns)
            },

            "schema_validation": {
                "expected_columns": self.expected_columns,
                "missing_columns": missing_columns,
                "unexpected_columns": unexpected_columns
            },

            "missing_value_analysis": {
                "missing_values": missing_value_report,
                "missing_percentage": missing_percentage,
                "allowed_missing_columns": self.optional_missing_columns
            },

            "duplicate_analysis": {
                "duplicate_rows": duplicate_rows,
                "duplicate_product_ids": duplicate_product_ids
            },

            "price_validation": {
                "negative_prices": negative_prices,
                "zero_prices": zero_prices,
                "non_finite_prices": non_finite_prices,
                "price_outliers": price_outliers
            },

            "data_type_validation": data_type_checks,

            "high_cardinality_columns": high_cardinality_columns,

            "validation_errors": validation_errors,

            "validation_warnings": validation_warnings,

            "validation_status": (
                "PASSED" if not validation_errors else "FAILED"
            )
        }

        logging.info(
            f"Validation status: "
            f"{validation_report['validation_status']}"
        )

        return validation_report

    def initiate_data_validation(self):

        logging.info("Entered Data Validation component")

        try:

            # Read ingested dataset
            df = pd.read_csv(
                self.validation_config.raw_data_path
            )

            logging.info("Dataset loaded successfully")

            # Perform validation
            validation_report = self.validate_dataset(df)

            # Create validation directory
            os.makedirs(
                os.path.dirname(
                    self.validation_config.validation_report_path
                ),
                exist_ok=True
            )

            # Save validation report
            with open(
                self.validation_config.validation_report_path,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    validation_report,
                    file,
                    indent=4
                )

            logging.info("Validation report saved successfully")

            if validation_report["validation_status"] == "FAILED":

                raise ValueError(
                    f"Dataset validation failed: "
                    f"{validation_report['validation_errors']}"
                )

            logging.info("Data Validation completed successfully")

            return self.validation_config.validation_report_path

        except Exception as e:

            raise CustomException(e, sys)


if __name__ == "__main__":

    obj = DataValidation()

    report_path = obj.initiate_data_validation()

    print("Validation report generated:", report_path)
