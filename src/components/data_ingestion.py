import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# from src.components.data_transformation import DataTransformation
# from src.components.data_transformation import DataTransformationConfig

# from src.components.model_trainer import ModelTrainerConfig
# from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:

    # Get project root directory
    project_root: str=os.path.abspath(os.path.join(os.path.dirname(__file__),"../.."))

    # Define data paths
    train_data_path: str=os.path.join(project_root,"artifacts","processed","train.csv")
    test_data_path: str=os.path.join(project_root,"artifacts","processed","test.csv")
    raw_data_path: str = os.path.join(project_root, "artifacts", "raw", "data.csv")

    source_data_path: str = os.path.join(project_root,"notebook","data","indian_pharmaceutical_products_clean.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered Data Ingestion method or component")
        try:
            df = pd.read_csv(self.ingestion_config.source_data_path)

            logging.info("Read the dataset as df")
            logging.info(f"Dataset shape: {df.shape}")

            # Create raw and processed directories
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            # Save raw dataset
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Raw dataset saved successfully")

            # Train test split
            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            # Save training dataset
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            # Save testing dataset
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Data Ingestion is completed")
            logging.info(f"Training data shape: {train_set.shape}")
            logging.info(f"Testing data shape: {test_set.shape}")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()

    print("Training data:", train_data)
    print("Testing data:", test_data)

            