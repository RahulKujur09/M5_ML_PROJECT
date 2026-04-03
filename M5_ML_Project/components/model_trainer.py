import os, sys
from M5_ML_Project.exception.exception import CustomException
from M5_ML_Project.logging.logger import logging
import pandas as pd
import numpy as np
from M5_ML_Project.entity.config_entity import ModelTrainingConfig
from M5_ML_Project.entity.artifact_entity import DataTransformationArtifacts, ModelTrainingArtifact
from M5_ML_Project.constant.training_pipeline import training_pipeline
from M5_ML_Project.utils.main_utils import evaluate_models, save_report, get_regression_report, save_object
from sklearn.linear_model import (LinearRegression, Ridge, Lasso)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

class ModelTraining:
    def __init__(self, model_training_config : ModelTrainingConfig) -> None:
        try:
            self.model_file_path : str = model_training_config.model_file_path

            self.report_file_path : str = model_training_config.report_file_path
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)
        
    @staticmethod
    def get_dataframe(file_path):
        try:
            return pd.read_csv(file_path, low_memory=False)
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)
        
    @staticmethod
    def get_score(r):
        try:
            scores = []
            for key, values in r.items():
                scores.append(values["rmse"])

            best_score = max(scores)

            return best_score
        except Exception as e:
            logging.error(e, sys)
            raise CustomException(e, sys)
        
    @staticmethod
    def get_model(r, score):
        try:
            for key, values in r.items():
                if values["rmse"] == score:
                    return values["model"]
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)
        
    def initiate_model_training(self, data_transformation_artifact : DataTransformationArtifacts) -> ModelTrainingArtifact:
        try:

            models = {
                        # "LinearRegression": LinearRegression(),
                        # "Ridge": Ridge(),
                        # "Lasso": Lasso(),
                        "XGBoost": XGBRegressor(tree_method = "auto"),
                        "LightGBM": LGBMRegressor(device = "gpu"),
                        "CatBoost": CatBoostRegressor(verbose=0, thread_count=-1, loss_function="RMSE", allow_writing_files=False)
                    }
            
            params = {
                        "LightGBM": {
                            "n_estimators": [200, 500],
                            "learning_rate": [0.05, 0.1],
                            "num_leaves": [31, 64]
                        },

                        "XGBoost": {
                            "n_estimators": [200, 500],
                            "max_depth": [6, 10],
                            "learning_rate": [0.05, 0.1]
                        },

                        "CatBoost": {
                            "iterations": [200],
                            "depth": [6, 8],
                            "learning_rate": [0.05]
                        }
                    }
            
            train_df = ModelTraining.get_dataframe(data_transformation_artifact.train_set_file_path)
            final_train_df = train_df.replace(["NAN", "nan", "Nan", "na", "Na", "NA"], np.nan)
            test_df = ModelTraining.get_dataframe(data_transformation_artifact.test_set_file_path)
            final_test_df = test_df.replace(["NAN", "nan", "Nan", "na", "Na", "NA"], np.nan)

            y_train = final_train_df[training_pipeline.TARGET_COLUMN]
            x_train = final_train_df.drop(columns=[training_pipeline.TARGET_COLUMN, "date", "sales"])

            y_test = final_test_df[training_pipeline.TARGET_COLUMN]
            x_test = final_test_df.drop(columns=[training_pipeline.TARGET_COLUMN, "date", "sales"])

            # x_train = x_train.to_numpy("float32")
            # x_test  = x_test.to_numpy("float32")
            # y_train = y_train.to_numpy("float32")
            # y_test  = y_test.to_numpy("float32")

            # After creating x_train, y_train
            sample_frac = 0.2 if len(x_train) > 500000 else 1.0
            if sample_frac < 1.0:
                sampled = x_train.sample(frac=sample_frac, random_state=42)
                x_train_small = sampled
                y_train_small = y_train.loc[sampled.index]
            else:
                x_train_small, y_train_small = x_train, y_train

            results = evaluate_models(x_train_small, y_train_small, models, params)

            best_model_name = min(results, key=lambda x: results[x][1])
            best_model = results[best_model_name][0]

            print("Best Model:", best_model_name)

            save_object(self.model_file_path, best_model)

            y_pred = best_model.predict(x_test)

            regression_report = get_regression_report(y_true=y_test, y_pred=y_pred)

            print(regression_report)

            return ModelTrainingArtifact(trained_model_file_path=self.model_file_path)


        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)