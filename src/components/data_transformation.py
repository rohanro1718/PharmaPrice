import os
import sys
import numpy as np
import pandas as pd
import ast
import re

from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationConfig:

    project_root: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )

    train_data_path: str = os.path.join(
        project_root, "artifacts", "processed", "train.csv"
    )

    test_data_path: str = os.path.join(
        project_root, "artifacts", "processed", "test.csv"
    )

    transformed_train_path: str = os.path.join(
    project_root, "artifacts", "processed", "engineered_train.csv"
    )

    transformed_test_path: str = os.path.join(
        project_root, "artifacts", "processed", "engineered_test.csv"
    )


class DataTransformation:

    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def engineer_manufacturer_features(self, df):

        logging.info("Starting manufacturer feature engineering")

        df["manufacturer_clean"] = (
            df["manufacturer_raw"]
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.title()
        )

        return df

    def engineer_packaging_features(self, df):

        logging.info("Starting packaging feature engineering")

        # 1. Extract package type
        df["package_type"] = (
            df["packaging_raw"]
            .str.extract(r"^(\w+)", expand=False)
            .str.lower()
        )

        # 2. Extract package quantity
        df["package_quantity"] = pd.to_numeric(
            df["packaging_raw"]
            .str.extract(
                r"\bof\s+([\d,.]+)",
                expand=False
            )
            .str.replace(",", "", regex=False),
            errors="coerce"
        )

        # 3. Identify records where pack size differs
        different_pack_sizes = df[
            df["pack_size"].notna()
            & (df["package_quantity"] != df["pack_size"])
        ].copy()

        # 4. Identify verified ratio-of-10 errors
        ratio = (
            different_pack_sizes["pack_size"]
            / different_pack_sizes["package_quantity"]
        )

        ratio_10_mask = (
            np.isfinite(ratio)
            & np.isclose(ratio, 10)
            & np.isclose(
                different_pack_sizes["pack_size"],
                different_pack_sizes["package_quantity"] * 10
            )
        )

        ratio_10_indices = different_pack_sizes.index[
            ratio_10_mask
        ]

        logging.info(
            f"Verified ratio-of-10 records: {len(ratio_10_indices)}"
        )

        # 5. Create final pack size
        df["pack_size_final"] = df["pack_size"]

        # Fill missing original pack sizes
        df["pack_size_final"] = (
            df["pack_size_final"].fillna(
                df["package_quantity"]
            )
        )

        # Correct only verified decimal-conversion errors (df.loc[row_condition, column_name])
        df.loc[
            ratio_10_indices,
            "pack_size_final" 
        ] = df.loc[
            ratio_10_indices,
            "package_quantity"
        ]

        # 6. Flag unresolved mismatches
        df["pack_size_mismatch"] = (
            df["pack_size"].notna()
            & (df["pack_size"] != df["package_quantity"])
            & (~df.index.isin(ratio_10_indices))
        )

        logging.info(
            f"Remaining packaging mismatches: "
            f"{df['pack_size_mismatch'].sum()}"
        )

        return df

    def parse_ingredients(self, value):
        try:
            if pd.isna(value):
                return []

            ingredients = ast.literal_eval(value)

            if not isinstance(ingredients, list):
                return []

            return [
                item for item in ingredients
                if isinstance(item, dict)
            ]

        except (ValueError, SyntaxError, TypeError):
            return []


    
    def parse_strength(self, value):

        # Handle missing values
        if pd.isna(value):
            return pd.Series({
                "strength_value": None,
                "strength_unit": None,
                "strength_type": "missing",
                "denominator_value": None,
                "denominator_unit": None
            })

        # Convert to lowercase and remove whitespace
        value = str(value).strip().lower()
        value = re.sub(r"\s+", "", value)

        # Handle malformed decimal points before units
        value = re.sub(
            r"(?<=\d)\.(?=(?:mg|gm|g|ml)/)",
            "",
            value
        )

        # 1. Concentration format: 50mg/5ml
        concentration = re.fullmatch(
            r"(\d+(?:\.\d+)?)(mg|gm|g|ml)/"
            r"(?:(\d+(?:\.\d+)?))?(ml|mg|gm|g)",
            value
        )

        if concentration:
            numerator, unit, denominator, denominator_unit = concentration.groups()

            return pd.Series({
                "strength_value": float(numerator),
                "strength_unit": "g" if unit in ["gm", "g"] else unit,
                "strength_type": "concentration",
                "denominator_value": float(denominator) if denominator else 1.0,
                "denominator_unit": (
                    "g" if denominator_unit in ["gm", "g"]
                    else denominator_unit
                )
            })

        # 2. Simple format: 500mg
        simple = re.fullmatch(
            r"(\d+(?:\.\d+)?)(mg|gm|g|ml)",
            value
        )

        if simple:
            number, unit = simple.groups()

            return pd.Series({
                "strength_value": float(number),
                "strength_unit": "g" if unit in ["gm", "g"] else unit,
                "strength_type": "simple",
                "denominator_value": None,
                "denominator_unit": None
            })

        # 3. Percentage format: 1%
        percentage = re.fullmatch(
            r"(\d+(?:\.\d+)?)%",
            value
        )

        if percentage:
            return pd.Series({
                "strength_value": float(percentage.group(1)),
                "strength_unit": "%",
                "strength_type": "percentage",
                "denominator_value": None,
                "denominator_unit": None
            })

        # 4. Unknown format
        return pd.Series({
            "strength_value": None,
            "strength_unit": None,
            "strength_type": "unknown",
            "denominator_value": None,
            "denominator_unit": None
        })


    def engineer_strength_features(self, df):

        logging.info("Starting strength feature engineering")

        strength_features = df["primary_strength"].apply(
            self.parse_strength
        )

        df[[
            "strength_value",
            "strength_unit",
            "strength_type",
            "denominator_value",
            "denominator_unit"
        ]] = strength_features

        logging.info("Strength feature engineering completed")

        return df


    def engineer_ingredient_features(self, df):

        logging.info("Starting ingredient feature engineering")

        # Parse active ingredients
        ingredient_records = df["active_ingredients"].apply(
            self.parse_ingredients
        )

        # Extract ingredient names
        ingredient_names = ingredient_records.apply(
            lambda records: [
                item["name"]
                for item in records
                if item.get("name")
            ]
        )

        # Extract ingredient strengths
        ingredient_strengths = ingredient_records.apply(
            lambda records: [
                item.get("strength")
                for item in records
            ]
        )

        # Create normalized ingredient combination
        df["ingredient_combination"] = ingredient_names.apply(
            lambda ingredients: " + ".join(
                sorted({
                    str(name).strip().casefold()
                    for name in ingredients
                    if str(name).strip()
                })
            )
        )

        logging.info("Ingredient feature engineering completed")

        return df

    def engineer_features(self, df):

        df = df.copy()

        df = self.engineer_manufacturer_features(df)

        df = self.engineer_packaging_features(df)

        df = self.engineer_ingredient_features(df)

        df = self.engineer_strength_features(df)

        return df

    def initiate_data_transformation(self):
        try:
            logging.info("Starting data transformation")

            train_df = pd.read_csv(
                self.transformation_config.train_data_path
            )

            test_df = pd.read_csv(
                self.transformation_config.test_data_path
            )

            train_df = self.engineer_features(train_df)
            test_df = self.engineer_features(test_df)

            os.makedirs(
                os.path.dirname(
                    self.transformation_config.transformed_train_path
                ),
                exist_ok=True
            )

            train_df.to_csv(
                self.transformation_config.transformed_train_path,
                index=False
            )

            test_df.to_csv(
                self.transformation_config.transformed_test_path,
                index=False
            )

            logging.info("Engineered datasets saved successfully")

            return (
                self.transformation_config.transformed_train_path,
                self.transformation_config.transformed_test_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":

    obj = DataTransformation()

    train_path, test_path = obj.initiate_data_transformation()

    print("\nEngineered datasets saved successfully!")
    print("Training data:", train_path)
    print("Testing data:", test_path)
