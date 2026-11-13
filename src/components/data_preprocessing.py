import os
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging

class FrequencyEncoder(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.frequency_maps = {}

    def fit(self, X, y=None):

        X = pd.DataFrame(X).copy()

        self.frequency_maps = {}

        for column in X.columns:

            frequencies = (
                X[column]
                .value_counts(normalize=True, dropna=False)
            )

            self.frequency_maps[column] = frequencies.to_dict()

        return self

    def transform(self, X):

        X = pd.DataFrame(X).copy()

        transformed_data = pd.DataFrame(index=X.index)

        for column in X.columns:

            transformed_data[column] = (
                X[column]
                .map(self.frequency_maps[column])
                .fillna(0.0)
            )

        return transformed_data.to_numpy()


    
@dataclass
class DataPreprocessingConfig:

    project_root: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )

    train_data_path: str = os.path.join(
        project_root, "artifacts", "processed", "engineered_train.csv"
    )

    test_data_path: str = os.path.join(
        project_root, "artifacts", "processed", "engineered_test.csv"
    )

    preprocessed_train_path: str = os.path.join(
        project_root, "artifacts", "processed", "preprocessed_train.csv"
    )

    preprocessed_test_path: str = os.path.join(
        project_root, "artifacts", "processed", "preprocessed_test.csv"
    )

    preprocessor_path: str = os.path.join(
        project_root, "artifacts", "models", "preprocessor.pkl"
    )

class DataPreprocessing:

    def __init__(self):
        self.preprocessing_config = DataPreprocessingConfig()

        # Target variable
        self.target_column = "price_inr"

        # Numerical features
        self.numerical_columns = [
            "num_active_ingredients",
            "pack_size_final",
            "strength_value"
        ]

        self.denominator_columns = [
            "denominator_value"
        ]

        self.indicator_columns = [
            "has_denominator",
            "strength_missing"
        ]

        # One-Hot Encoding features
        self.onehot_columns = [
            "is_discontinued",
            "dosage_form",
            "pack_unit",
            "therapeutic_class",
            "package_type",
            "strength_unit",
            "strength_type",
            "denominator_unit"
        ]

        # Frequency Encoding features
        self.frequency_columns = [
            "primary_ingredient",
            "ingredient_combination",
            "manufacturer_clean"
        ]

    def create_missing_indicators(self, df):

        df = df.copy()

        df["has_denominator"] = (
            df["denominator_value"].notna().astype(int)
        )

        df["strength_missing"] = (
            df["strength_value"].isna().astype(int)
        )

        return df

    def create_numerical_pipeline(self):

        numerical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        keep_empty_features=True
                    )
                ),
                ("scaler", StandardScaler())
            ]
        )

        return numerical_pipeline

    def create_denominator_pipeline(self):

        denominator_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value=1.0
                    )
                ),
                ("scaler", StandardScaler())
            ]
        )

        return denominator_pipeline

    
    def create_categorical_pipeline(self):

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value="Missing"
                    )
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]
        )

        return categorical_pipeline

    
    def create_frequency_pipeline(self):

        frequency_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value="Missing"
                    )
                ),
                (
                    "frequency_encoder",
                    FrequencyEncoder()
                )
            ]
        )

        return frequency_pipeline


    
    def create_preprocessor(self):

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numerical",
                    self.create_numerical_pipeline(),
                    self.numerical_columns
                ),
                (
                    "denominator",
                    self.create_denominator_pipeline(),
                    self.denominator_columns
                ),
                (
                    "categorical",
                    self.create_categorical_pipeline(),
                    self.onehot_columns
                ),
                (
                    "frequency",
                    self.create_frequency_pipeline(),
                    self.frequency_columns
                ),
                (
                    "indicators",
                    "passthrough",
                    self.indicator_columns
                )
            ]
        )

        return preprocessor


