from src.logging.logger import logging
from src.exception.exception import CustomException
from src.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
from src.component.data_ingestion import DataIngestion
from src.component.data_validation import DataValidation
from src.component.data_transformation import DataTransformation
from src.component.model_trainer import ModelTrainer
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
        logging.info("Data Transformation begining")
        data_transformation_config = DataTransformationConfig(trainingpipeline)
        data_transformation = DataTransformation(data_transformation_config, data_validation_artifacts)
        data_transformation_artifacts = data_transformation.initiate_data_transformation()
        logging.info("Data Transformation complete")
        logging.info("Starting Model training")
        model_trainer_config = ModelTrainerConfig(trainingpipeline)
        model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifacts)
        results = model_trainer.initiate_data_trainer()
        logging.info("Model training complete")


    except Exception as e:
        raise CustomException(e, sys.exc_info())