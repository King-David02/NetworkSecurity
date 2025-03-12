import os
import sys
import pandas as pd
import pymongo
from dotenv import load_dotenv
from src.exception.exception import CustomException
from src.logging.logger import logging

load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")

class DataExtract:
    def __init__(self, mongo_url):
        self.mongo_client = pymongo.MongoClient(mongo_url)

    def csv_to_json(self, file_path):
        data = pd.read_csv(file_path)
        records = data.to_dict(orient='records')
        return records
    
    def insert_to_mongo(self,records, database, collection):
        db = self.mongo_client[database]
        coll = db[collection]
        coll.insert_many(records)
        return len(records)
    
    def close_connection(self):
        self.mongo_client.close()


if __name__ == "__main__":
    extractor = DataExtract(MONGO_DB_URL)
    FILE_PATH = os.path.join("Network_Data", "phisingData.csv")
    DATABASE = "KING-DAVID"
    COLLECTION = "Network Data"

    try:
        records = extractor.csv_to_json(FILE_PATH)
        logging.info("Records saved as JSON")
        records_counts = extractor.insert_to_mongo(records, DATABASE, COLLECTION,)
        logging.info(f"Number of records saved: {records_counts}")

    except Exception as e:
        raise CustomException(e, sys.exc_info())
    
    finally:
        extractor.close_connection()