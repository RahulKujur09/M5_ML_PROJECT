import os, sys
import pandas as pd
import yaml
import pickle
from M5_ML_Project.logging.logger import logging
from M5_ML_Project.exception.exception import CustomException
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_squared_error

def get_schema(file_path):
    try:
        with open(file_path, "r") as file_obj:
            col=yaml.safe_load(file_obj)
            column = list(col["columns"].keys())
            return column
    except Exception as e:
        logging.error(e)
        raise CustomException(e, sys)

def save_report(content, file_path):
    try:
        with open(file_path, "w") as file_obj:
            yaml.dump(content, file_obj)
    except Exception as e:
        logging.error(e)
        raise CustomException(e, sys)
    
def get_model_report(model_grid, x_train, y_train, y_test, x_test):
    try:
        report = {}

        for model_name, (model, param) in model_grid.items():
            gs = RandomizedSearchCV(
                estimator=model,
                param_distributions=param,
                n_iter=10,
                scoring="neg_root_mean_squared_error",
                cv=TimeSeriesSplit(n_splits=3),
                n_jobs=1,
                pre_dispatch=1,
                random_state=42
            )

            gs.fit(X=x_train, y=y_train)

            model_trained = gs.best_estimator_
            params = gs.best_params_
            score = root_mean_squared_error(y_true=y_test, y_pred=gs.predict(X=x_test))

            report[model_name] = {
                "model" : model_trained,
                "params" : params,
                "rmse" : score
            }
        
        return report
    except Exception as e:
        logging.error(e)
        raise CustomException(e, sys)
    
def get_regression_report(y_true, y_pred):
    try:
        return pd.DataFrame({
            "mean_absolute_error" : mean_absolute_error(y_pred=y_pred, y_true=y_true),
            "mean_squared_error" : mean_squared_error(y_pred=y_pred, y_true=y_true),
            "root_mean_squared_error" : root_mean_squared_error(y_pred=y_pred, y_true=y_true)
        })
    except Exception as e:
        logging.error(e)
        raise CustomException(e, sys)
    
def save_object(file_path, content):
    try:
        with open(file_path, "wb") as file_obj:
            pickle.dump(content, file_obj)
    except Exception as e:
        logging.error(e)
        raise CustomException(e, sys)