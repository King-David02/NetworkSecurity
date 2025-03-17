from datetime import datetime
import os
from src.constant import training_pipeline as tp

class TrainingPipelineConfig:
    def __init__(self, timestamp= datetime.now()):
        timestamp = timestamp.strftime("%d_%m_%Y_%H-%M-%S")
        self.pipeline_name = tp.PIPELINE_NAME
        self.artifacts_dir = os.path.join(tp.ARTIFACT_DIR, timestamp)
        self.model_dir=os.path.join("final_model")

    
class DataIngestionConfig:
    def __init__(self, ingestion_config: TrainingPipelineConfig):
        self.ingestion_base_dir = os.path.join(ingestion_config.artifacts_dir, tp.DATA_INGESTION_DIR_NAME)
        self.feature_store = os.path.join(self.ingestion_base_dir, tp.DATA_INGESTION_FEATURE_STORE_DIR, tp.FILE_NAME)
        self.ingested_dir = os.path.join(self.ingestion_base_dir, tp.DATA_INGESTION_INGESTED_DIR)
        self.train_file_path = os.path.join(self.ingested_dir, tp.TRAIN_FILE_NAME)
        self.test_file_path = os.path.join(self.ingested_dir, tp.TEST_FILE_NAME)
        self.train_test_split_ratio = tp.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        self.collection_name = tp.DATA_INGESTION_COLLECTION_NAME
        self.database_name = tp.DATA_INGESTION_DATABASE_NAME


class DataValidationConfig:
    def __init__(self, validation_config: TrainingPipelineConfig):
        self.validation_base_dir = os.path.join(validation_config.artifacts_dir, tp.DATA_VALIDATION_DIR_NAME)
        self.valid_dir = os.path.join(self.validation_base_dir, tp.DATA_VALIDATION_VALID_DIR)
        self.invalid_dir = os.path.join(self.validation_base_dir, tp.DATA_VALIDATION_INVALID_DIR)
        self.drift_report_dir = os.path.join(self.validation_base_dir, tp.DATA_VALIDATION_DRIFT_REPORT_DIR, tp.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)
        self.valid_train_file_path= os.path.join(self.valid_dir, tp.TRAIN_FILE_NAME)
        self.valid_test_file_path = os.path.join(self.valid_dir, tp.TEST_FILE_NAME)
        self.invalid_train_file_path= os.path.join(self.invalid_dir, tp.TRAIN_FILE_NAME)
        self.invalid_test_file_path = os.path.join(self.invalid_dir, tp.TEST_FILE_NAME)



class DataTransformationConfig:
    def __init__(self, transformation_config: TrainingPipelineConfig):
        self.transformation_base_dir = os.path.join(transformation_config.artifacts_dir, tp.DATA_TRANSFORMATION_DIR_NAME)
        self.transformed_data = os.path.join(self.transformation_base_dir, tp.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR)
        self.transformed_object = os.path.join(self.transformation_base_dir, tp.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR, tp.PREPROCESSING_OBJECT_FILE_NAME)
        self.transformed_train_file_path = os.path.join(self.transformed_data, tp.DATA_TRANSFORMATION_TRAIN_FILE_PATH)
        self.transformed_test_file_path = os.path.join(self.transformed_data, tp.DATA_TRANSFORMATION_TEST_FILE_PATH)


class ModelTrainerConfig:
    def __init__(self, trainer_config: TrainingPipelineConfig):
        self.trainer_base_dir = os.path.join(trainer_config.artifacts_dir, tp.MODEL_TRAINER_DIR_NAME)
        self.saved_model_path = os.path.join(self.trainer_base_dir, tp.MODEL_TRAINER_TRAINED_MODEL_NAME)