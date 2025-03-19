import pandas as pd
import numpy as np
import sys
import os
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifacts_entity import DataValidationArtifact, DataTransformationArtifact
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from src.constant.training_pipeline import TARGET_COLUMN, DATA_TRANSFORMATION_IMPUTER_PARAMS
from src.exception.exception import CustomException
from src.logging.logger import logging
from src.utils.main_utils.utils import save_numpy_array_data, save_object


class DataTransformation:
    def __init__(self, config: DataTransformationConfig, artifacts: DataValidationArtifact):
        self.config = config
        self.artifacts = artifacts

    def read_data_frame(self, file_path):
        df = pd.read_csv(file_path)
        return df


    def transformation(self):
        imputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
        return Pipeline([("imputer", imputer)])
    

    def initiate_data_transformation(self):
        try:
            train_df = self.read_data_frame(self.artifacts.valid_train_file_path)
            test_df = self.read_data_frame(self.artifacts.valid_test_file_path)
            logging.info("Train and test data received")

            X_train = train_df.drop(columns=[TARGET_COLUMN])
            y_train = train_df[TARGET_COLUMN].replace(-1, 0)
            X_test = test_df.drop(columns=[TARGET_COLUMN])
            y_test = test_df[TARGET_COLUMN].replace(-1, 0)

            logging.info("Starting Preprocessing")
            preprocessing = self.transformation()
            X_train_transformed = preprocessing.fit_transform(X_train)
            X_test_transformed = preprocessing.transform(X_test)
            logging.info("preprocessing complete")

            train_arr = np.c_[X_train_transformed, y_train]
            test_arr = np.c_[X_test_transformed, y_test]

            save_numpy_array_data(self.config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.config.transformed_test_file_path, test_arr)
            logging.info("Train and Test array Saved")
            save_object(self.config.transformed_object, preprocessing)
            save_object('final_objects/preprocessing.pkl', preprocessing)
            logging.info("preprocessing object saved")

            artifacts = DataTransformationArtifact(transformed_object_file_path=self.config.transformed_object, transformed_train_file_path=self.config.transformed_train_file_path,
                                                transformed_test_file_path=self.config.transformed_test_file_path)
            return artifacts
        
        except Exception as e:
            raise CustomException(e, sys.exc_info())