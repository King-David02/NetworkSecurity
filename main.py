from src.logging.logger import logging
from src.exception.exception import CustomException
from src.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig
from src.component.data_ingestion import DataIngestion
from src.component.data_validation import DataValidation
import sys

if __name__ == "__main__":
    try:
        logging.info("Data ingestion begining")
        trainingpipeline = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(trainingpipeline)
        data_ingestion= DataIngestion(data_ingestion_config)
        data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()
        logging.info(f'{data_ingestion_artifacts}')
        logging.info("data ingestion completed")
        logging.info("Starting validation from main")
        data_validation_config = DataValidationConfig(trainingpipeline)
        data_validation = DataValidation(data_validation_config, data_ingestion_artifacts)
        data_validation_artifacts = data_validation.initiate_data_validation()
        logging.info(f"{data_validation_artifacts}")
        logging.info("Data_validation completed")



    except Exception as e:
        raise CustomException(e, sys.exc_info())