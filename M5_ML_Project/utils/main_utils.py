import os, sys
import pandas as pd
import numpy as np
import yaml
import pickle
from M5_ML_Project.logging.logger import logging
from M5_ML_Project.exception.exception import CustomException
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV

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
    
from sklearn.model_selection import RandomizedSearchCV

def evaluate_models(X_train, y_train, models, params):

    results = {}

    kf = KFold(n_splits=2, shuffle=True, random_state=42)  # reduce folds

    for name, model in models.items():

        print(f"\nTraining {name}")

        if name in params:

            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=params[name],
                n_iter=2,                # ⭐ VERY IMPORTANT
                cv=kf,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,                # ⭐ MEMORY SAFE
                verbose=1,
                random_state=42
            )

            search.fit(X_train, y_train)

            best_model = search.best_estimator_
            score = -search.best_score_

        else:
            model.fit(X_train, y_train)

            preds = model.predict(X_train)
            score = np.sqrt(mean_squared_error(y_train, preds))
            best_model = model

        results[name] = (best_model, score)

        print(f"{name} RMSE:", score)

    return results
    
def get_regression_report(y_true, y_pred):
    try:
        return pd.DataFrame({
            "mean_absolute_error" : [mean_absolute_error(y_pred=y_pred, y_true=y_true)],
            "mean_squared_error" : [mean_squared_error(y_pred=y_pred, y_true=y_true)],
            "root_mean_squared_error" : [root_mean_squared_error(y_pred=y_pred, y_true=y_true)]
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