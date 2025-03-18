import sys
from src.entity.artifacts_entity import(
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact
)
from src.entity.config_entity import(
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)
from src.component.data_ingestion import DataIngestion
from src.component.data_transformation  import DataTransformation
from src.component.data_validation import DataValidation
from src.component.model_trainer import ModelTrainer
from src.entity.config_entity import TrainingPipelineConfig
from src.exception.exception import CustomException
from src.logging.logger import logging

class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config=TrainingPipelineConfig()

    def start_data_ingestion(self):
        try:
            data_ingestion_config = DataIngestionConfig(self.training_pipeline_config)
            data_ingestion = DataIngestion(data_ingestion_config)
            data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()
            return data_ingestion_artifacts
        
        except Exception as e:
            raise CustomException(e, sys.exc_info())

    def start_data_validation(self, artifacts:DataIngestionArtifact):
        try:
            data_validation_config = DataValidationConfig(self.training_pipeline_config)
            data_validation = DataValidation(config=data_validation_config, artifacts=artifacts)
            data_validation_artifacts = data_validation.initiate_data_validation()
            return data_validation_artifacts
        
        except Exception as e:
            raise CustomException(e, sys.exc_info())
    

    def start_data_transformation(self, artifacts:DataValidationArtifact):
        try:
            data_transformaton_config = DataTransformationConfig(self.training_pipeline_config)
            data_transformaton = DataTransformation(config=data_transformaton_config, artifacts=artifacts)
            data_validation_artifacts = data_transformaton.initiate_data_transformation()
            return data_validation_artifacts
        
        except Exception as e:
            raise CustomException(e, sys.exc_info())
    
    def start_model_trainer(self, data_validation_artifacts: DataValidationArtifact):
        try:
            model_trainer_config = ModelTrainerConfig(self.training_pipeline_config)
            model_trainer = ModelTrainer(config=model_trainer_config, artifacts=data_validation_artifacts)
            model_trainer_artifacts = model_trainer.initiate_data_trainer()
            return model_trainer_artifacts
        
        except Exception as e:
            raise CustomException(e, sys.exc_info())
    
    def run_pipeline(self):
        try:
            data_ingestion_artifacts = self.start_data_ingestion()
            data_validation_artifacts = self.start_data_validation(data_ingestion_artifacts)
            data_transformation_artifacts = self.start_data_transformation(data_validation_artifacts)
            model_trainer_artifacts = self.start_model_trainer(data_transformation_artifacts)
            return model_trainer_artifacts
        
        except Exception as e:
            raise CustomException(e, sys.exc_info())