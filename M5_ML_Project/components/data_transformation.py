import os, sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from M5_ML_Project.logging.logger import logging
from M5_ML_Project.exception.exception import CustomException
from M5_ML_Project.entity.config_entity import DataTransformationConfig
from M5_ML_Project.entity.artifact_entity import DataValidationartifacts, DataTransformationArtifacts

class DataTransformation:
    def __init__(self, data_transformation_config : DataTransformationConfig):
        try:
            self.train_file_path : str = data_transformation_config.train_file_name

            self.test_file_path : str = data_transformation_config.test_file_name
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
    def reduce_mem_usage(df : pd.DataFrame):
        try:
            for col in df.columns:
                col_type = df[col].dtype

                if col_type != "object":
                    col_min = df[col].min()
                    col_max = df[col].max()

                    if str(col_type)[:3] == "int":
                        if col_min >= 0:
                            df[col] = df[col].astype("int32")
                        else:
                            df[col] = df[col].astype("int32")
                else:
                    df[col] = df[col].astype("float")
            return df
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)
        
    @staticmethod
    def label_encoding(df : pd.DataFrame):
        try:

            df["sell_price"] = df.groupby(["store_id", "item_id"])["sell_price"].ffill().bfill()

            cat_cols = list(df.select_dtypes("str").columns)

            for col in cat_cols:
                df[col] = LabelEncoder().fit_transform(
                    df[col].astype("str")
                )
            
            return df
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)
        
    @staticmethod
    def cat_col_transformer(df : pd.DataFrame):
        try:
            df["date"] = pd.to_datetime(df["date"])

            df["day"] = df["date"].dt.day
            df["week"] = df["date"].dt.isocalendar().week.astype("int16")
            df["month"] = df["date"].dt.month
            df["year"] = df["date"].dt.year
            df["dayofweek"] = df["date"].dt.dayofweek
            df["is_weekend"] = (df["dayofweek"] >=5).astype("int8")

            return df
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, data_validation_artifact: DataValidationartifacts):
        try:
            train_df = DataTransformation.get_dataframe(data_validation_artifact.valid_train_set_file_path)

            test_df = DataTransformation.get_dataframe(data_validation_artifact.valid_test_set_file_path)

            train_df_num_col_transformed = DataTransformation.reduce_mem_usage(train_df)

            test_df_num_col_transformed = DataTransformation.reduce_mem_usage(test_df)



            final_train_df = DataTransformation.cat_col_transformer(train_df_num_col_transformed)

            final_test_df = DataTransformation.cat_col_transformer(test_df_num_col_transformed)

            train_df_level_encoding = DataTransformation.label_encoding(final_train_df)

            test_df_level_encoding = DataTransformation.label_encoding(final_test_df)

            

            train_df_level_encoding.to_csv(self.train_file_path, index=False)

            test_df_level_encoding.to_csv(self.test_file_path, index=False)

            logging.info("data transformation completed")

            return DataTransformationArtifacts(train_set_file_path=self.train_file_path, test_set_file_path=self.test_file_path)
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)