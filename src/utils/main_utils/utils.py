import yaml
from src.exception.exception import CustomException
from src.logging.logger import logging
import os,sys
import numpy as np
#import dill
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score,precision_score,recall_score
from src.entity.artifacts_entity import ClassificationMetricArtifact


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CustomException(e, sys.exc_info())
    
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise CustomException(e, sys.exc_info())
    


def save_numpy_array_data(file_path: str, array: np.array):
        """
        Save numpy array data to file
        file_path: str location of file to save
        array: np.array data to save
        """
        try:
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)
            with open(file_path, "wb") as file_obj:
                np.save(file_obj, array)
        except Exception as e:
            raise CustomException(e, sys.exc_info())
        
def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys.exc_info()) from e
        

def save_object(file_path: str, obj: object) -> None:
        try:
            logging.info("Entered the save_object method of MainUtils class")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as file_obj:
                pickle.dump(obj, file_obj)
            logging.info("Exited the save_object method of MainUtils class")
        except Exception as e:
            raise CustomException(e, sys.exc_info())
        

def load_object(file_path: str, ) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys.exc_info())


def evaluate_model(X_train, X_test, y_train, y_test, model_params):
    scores = {}
    best_model_param ={}
    for name, (model, param) in model_params.items():
        gs = GridSearchCV(model, param, cv=2)
        gs.fit(X_train, y_train)
        best_params = gs.best_estimator_
        y_pred = best_params.predict(X_train)
        score = f1_score(y_train, y_pred)
        scores[name] = score
        best_model_param[name] = best_params

    best_model_name = max(scores, key=scores.get)
    best_model = best_model_param[best_model_name]
    best_score = scores[best_model_name]

    return best_model, best_score


def get_classification_score(y_true,y_pred)->ClassificationMetricArtifact:
    try:
            
        model_f1_score = f1_score(y_true, y_pred)
        model_recall_score = recall_score(y_true, y_pred)
        model_precision_score=precision_score(y_true,y_pred)

        classification_metric =  ClassificationMetricArtifact(f1_score=model_f1_score,
                    precision_score=model_precision_score, 
                    recall_score=model_recall_score)
        return classification_metric
    except Exception as e:
        raise CustomException(e,sys.exc_info())

