import os, sys
from M5_ML_Project.exception.exception import CustomException
from M5_ML_Project.logging.logger import logging
import pandas as pd
import numpy as np
from M5_ML_Project.entity.config_entity import ModelTrainingConfig
from M5_ML_Project.entity.artifact_entity import DataTransformationArtifacts, ModelTrainingArtifact
from M5_ML_Project.constant.training_pipeline import training_pipeline
from M5_ML_Project.utils.main_utils import trainlgbm, save_report, get_regression_report, save_object
from sklearn.linear_model import (LinearRegression, LogisticRegression, ridge_regression)
from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor)
from sklearn.tree import DecisionTreeRegressor

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
            model_name = []
            for key, values in r.items():
                if values["rmse"] == score:
                    model_name.append(key)

            best_model = r[model_name]["model"]

            return best_model
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)
        
    def initiate_model_training(self, data_transformation_artifact : DataTransformationArtifacts) -> ModelTrainingArtifact:
        try:

            model_grid = {
                "random_forest" : (RandomForestRegressor(), training_pipeline.MODEL_TRAINER_RANDOM_FOREST_REGRESSION_PARAMS),
                "gradient_boosting_regression" : (GradientBoostingRegressor(), training_pipeline.MODEL_TRAINER_GRADIENT_BOOSTING_REGRESSION_PARAMS),
                "decision_tree_regression" : (DecisionTreeRegressor(), training_pipeline.MODEL_TRAINER_DECISION_TREE_REGRESSION_PARAMS)
            }

            train_df = ModelTraining.get_dataframe(data_transformation_artifact.train_set_file_path)
            final_train_df = train_df.replace(["NAN", "nan", "Nan", "na", "Na", "NA"], np.nan)
            test_df = ModelTraining.get_dataframe(data_transformation_artifact.test_set_file_path)
            final_test_df = test_df.replace(["NAN", "nan", "Nan", "na", "Na", "NA"], np.nan)

            y_train = final_train_df[training_pipeline.TARGET_COLUMN]
            x_train = final_train_df.drop(columns=[training_pipeline.TARGET_COLUMN, "date"])

            y_test = final_test_df[training_pipeline.TARGET_COLUMN]
            x_test = final_test_df.drop(columns=[training_pipeline.TARGET_COLUMN, "date"])

            x_train = x_train.to_numpy("float32")
            x_test  = x_test.to_numpy("float32")
            y_train = y_train.to_numpy("float32")
            y_test  = y_test.to_numpy("float32")

            model = trainlgbm(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

            save_object(self.model_file_path, model)

            return ModelTrainingArtifact(trained_model_file_path=self.model_file_path)


        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)