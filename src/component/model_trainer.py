import pandas as pd
import numpy as np
import sys
import os
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifacts_entity import ModelTrainerArtifact, DataTransformationArtifact
from src.exception.exception import CustomException
from src.logging.logger import logging
from sklearn.neighbors import KNeighborsClassifier
from src.utils.main_utils.utils import load_numpy_array_data, save_object, evaluate_model, get_classification_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig, artifacts: DataTransformationArtifact):
        self.config = config
        self.artifacts = artifacts

    
    def initiate_data_trainer(self):
        try:
            models = {
            "Random Forest": (RandomForestClassifier(verbose=1), {
                "n_estimators": [8, 16, 32, 128, 256]
            }),
            "Decision Tree": (DecisionTreeClassifier(), {
                "criterion": ["gini", "entropy", "log_loss"]
            }),
            "Gradient Boosting": (GradientBoostingClassifier(verbose=1), {
                "learning_rate": [0.1, 0.01, 0.05, 0.001],
                "subsample": [0.6, 0.7, 0.75, 0.85, 0.9],
                "n_estimators": [8, 16, 32, 64, 128, 256]
            }),
            "Logistic Regression": (LogisticRegression(verbose=1), {}),
            "AdaBoost": (AdaBoostClassifier(), {
                "learning_rate": [0.1, 0.01, 0.001],
                "n_estimators": [8, 16, 32, 64, 128, 256]
            })
        }
            

            train_arr = load_numpy_array_data(self.artifacts.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.artifacts.transformed_test_file_path)

            X_train, y_train = train_arr[:,:-1], train_arr[:, -1]
            X_test, y_test = test_arr[:,:-1], test_arr[:, -1]

            logging.info("Model Training Begins")
            best_model, R2 = evaluate_model(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, model_params=models)
            logging.info("Model Training complete")
            logging.info(f"The best_model is {best_model} with R2 score of {R2}")
            save_object(self.config.saved_model_path, best_model)
            logging.info("Best Model Saved")
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_metrics = get_classification_score(y_train, y_train_pred)
            test_metrics = get_classification_score(y_test, y_test_pred)
            artifacts = ModelTrainerArtifact(trained_model_file_path=self.config.saved_model_path, train_metric_artifact=train_metrics,test_metric_artifact=test_metrics)
            return artifacts
        
        except Exception as e:
            raise CustomException(e, sys.exc_info())