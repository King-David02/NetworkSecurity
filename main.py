from src.logging.logger import logging
from src.exception.exception import CustomException
from src.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig
from src.component.data_ingestion import DataIngestion
import sys

if __name__ == "__main__":
    try:
        logging.info("Data ingestion begining")
        trainingpipeline = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(trainingpipeline)
        data_ingestion= DataIngestion(data_ingestion_config)
        artifacts = data_ingestion.initiate_data_ingestion()
        logging.info(f'{artifacts}')
        logging.info("data ingestion completed")


    except Exception as e:
        raise CustomException(e, sys.exc_info())