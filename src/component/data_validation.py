from src.exception.exception import CustomException
from src.logging.logger import logging
from src.entity.artifacts_entity import DataIngestionArtifact
from src.entity.config_entity import DataValidationConfig
from src.entity.artifacts_entity import DataValidationArtifact
from src.utils.main_utils.utils import read_yaml_file, write_yaml_file
from src.constant.training_pipeline import SCHEMA_FILE_PATH
import pandas as pd
import os
import sys
from scipy.stats import ks_2samp

class DataValidation:
    def __init__(self, config: DataValidationConfig, artifacts: DataIngestionArtifact):
        self.config = config
        self.artifacts = artifacts
        self.schema = read_yaml_file(SCHEMA_FILE_PATH)

    def read_csv(self, file_path):
        df = pd.read_csv(file_path)
        return df

    def check_columns(self, df):
        expected_columns = len(self.schema['columns'])
        actual_columns = len(df.columns)
        logging.info(f'{expected_columns} number of columns expected, {actual_columns} received')

    def checking_drift(self, train_df: pd.DataFrame, test_df: pd.DataFrame, threshold: float):
        no_drift = True
        report ={}
        for col in train_df.columns:
            stat = ks_2samp(train_df[col], test_df[col])
            drifted = bool(stat.pvalue < threshold)
            if drifted:
                no_drift = False
            report[col] = {"p_value": float(stat.pvalue), 
                           "drifted":drifted}

        report_dir_name = os.path.dirname(self.config.drift_report_dir)
        os.makedirs(report_dir_name, exist_ok=True)
        write_yaml_file(self.config.drift_report_dir, report)
        return no_drift
    
    def save_as_data_frame(self, df, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)


    def initiate_data_validation(self):
        try:
            logging.info("Data validation started")
            train_df = self.read_csv(self.artifacts.trained_file_path)
            test_df = self.read_csv(self.artifacts.test_file_path)
            logging.info("Train and Test Data received")
            logging.info("Comparing data columns to schema")
            self.check_columns(train_df)
            self.check_columns(test_df)

            logging.info("Checking Drift")
            drift = self.checking_drift(train_df, test_df, 0.05)

            logging.info("Saving Valid train and test df")
            os.makedirs(os.path.dirname(self.config.valid_train_file_path), exist_ok=True)
            self.save_as_data_frame(train_df, self.config.valid_train_file_path)
            self.save_as_data_frame(test_df, self.config.valid_test_file_path)


            Artifacts = DataValidationArtifact(validation_status=drift, valid_train_file_path=self.config.valid_train_file_path, valid_test_file_path=self.config.valid_test_file_path,
                                            invalid_train_file_path=None, invalid_test_file_path=None, drift_report_file_path=self.config.drift_report_dir)
            logging.info("Data validation complete")
            return Artifacts

        except Exception as e:
            raise CustomException(e, sys.exc_info())