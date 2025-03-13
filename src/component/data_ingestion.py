import os
import sys
import pymongo
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from src.entity.artifacts_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig
from sklearn. model_selection import train_test_split
from src.exception.exception import CustomException
from src.logging.logger import logging


load_dotenv()
MONGO_DB_URL = os.getenv('MONGO_DB_URL')

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)

    def retrieve_data_from_mongo(self):
        try:
            collection = self.mongo_client[self.config.database_name][self.config.collection_name]
            df = pd.DataFrame(list(collection.find())).drop(columns=["_id"], errors='ignore')
            df.replace({"na": pd.NA}, inplace=True)
            logging.info(f"Retrieved {len(df)} rows from MongoDB")
            return df
        
        except Exception as e:
            raise CustomException(e, sys.exc_info())
    
    def save_to_dataframe(self, df, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)


    def initiate_data_ingestion(self):
        try:
            logging.info("Data ingestion started")
            df = self.retrieve_data_from_mongo()
            logging.info("Datareceived from Mongodb")
            self.save_to_dataframe(df, self.config.feature_store)
            logging.info("Raw file Saved")

            train_df, test_df = train_test_split(df, test_size=self.config.train_test_split_ratio, random_state=42)
            logging.info("train test split done")
            self.save_to_dataframe(train_df, self.config.train_file_path)
            logging.info("train data saved")
            self.save_to_dataframe(test_df, self.config.test_file_path)
            logging.info("test data saved")
            artifacts = DataIngestionArtifact(trained_file_path=self.config.train_file_path, test_file_path=self.config.test_file_path)
            logging.info("Data ingestion done")
            return artifacts
        
        except Exception as e:
            raise CustomException(e, sys.exc_info())
    