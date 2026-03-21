import os, sys
import pandas as pd
import numpy as np
import yaml
import pickle
from M5_ML_Project.logging.logger import logging
from M5_ML_Project.exception.exception import CustomException
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_squared_error

import lightgbm as lgb

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
        
def rmse(y_true, y_predict):
    try:
        return np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_predict))
    except Exception as e:
        logging.error(e)
        raise CustomException(e, sys)
    
def trainlgbm(x_train, y_train, x_test, y_test):
    try:
        params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": 42,
        "verbosity": -1
    }

        train_data = lgb.Dataset(x_train, label=y_train)
        valid_data = lgb.Dataset(x_test, label=y_test)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=2000,
            valid_sets=[valid_data],
            callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(100)
        ]
    )

        preds = model.predict(x_test)
        score = rmse(y_test, preds)

        print("Validation RMSE:", score)

        return model
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