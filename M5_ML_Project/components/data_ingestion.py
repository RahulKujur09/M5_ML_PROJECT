import os, sys
import pandas as pd
import numpy as np
from M5_ML_Project.exception.exception import CustomException
from M5_ML_Project.logging.logger import logging
from M5_ML_Project.entity.config_entity import DataIngestionConfig
from M5_ML_Project.constant.training_pipeline import training_pipeline
from M5_ML_Project.entity.artifact_entity import DataIngestionArtifact

class DataIngestion:
    def __init__(self, data_ingestion_config : DataIngestionConfig):

        try:
            self.feature_store_file_name : str = data_ingestion_config.feature_store_file_name

            self.train_set_file_name : str = data_ingestion_config.train_file_name

            self.test_set_file_name : str = data_ingestion_config.test_file_name
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)

    @staticmethod
    def get_raw_csv(file_path):
        try:
            data_type ={
                "item_id" : "category",
                "dept_id" : "category",
                "cat_id" : "category",
                "store_id" : "category",
                "state_id" : "category"
            }
            return pd.read_csv(
                file_path,
                dtype=data_type,
                low_memory=False
                )
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)
        
    def initiate_data_ingestion(self, sales_train_validation_file_path, sell_prices_file_path, calendar_file_path) -> DataIngestionArtifact:
        
        try:
            self.sales_train_validation_df = DataIngestion.get_raw_csv(sales_train_validation_file_path)
            logging.info(f"fetched sales_train_validation as dataframe")

            self.sell_prices_df = DataIngestion.get_raw_csv(sell_prices_file_path)
            logging.info(f"fetched sell_price as dataframe")

            self.calendar_df = DataIngestion.get_raw_csv(calendar_file_path)
            logging.info(f"fetched calendar as dataframe")

            self.sales_train_validation_df = self.sales_train_validation_df[self.sales_train_validation_df["store_id"] == "CA_1"]

            melt_columns = [var for var in self.sales_train_validation_df.columns if var.startswith("d_")]

            col = self.sales_train_validation_df.columns.to_list()

            id_verse = []

            for i in col:
                if i.startswith("d_"):
                    col.remove(i)
                else:
                    id_verse.append(i)

            sales_train_validation_melted_df = self.sales_train_validation_df.melt(
                id_vars=id_verse,
                value_vars=melt_columns,
                value_name="sales",
                var_name="d"
            )
            logging.info(f"melted the sales_train_validation dataframe")

            sales_train_validation_melted_df_with_calendar = sales_train_validation_melted_df.merge(self.calendar_df, how="left", on="d")
            logging.info(f"merged sales_train_validation and calender dataframes")

            # sales_train_validation_melted_df_with_calendar_for_CA_1_store = sales_train_validation_melted_df_with_calendar[sales_train_validation_melted_df["store_id"] == "CA_1"]

            logging.info(f"seperated data from 'CA_1' store from 'sales_train_validation_melted_df_with_calendar'")

            sale_prices_df_CA_1 = self.sell_prices_df[self.sell_prices_df["store_id"] == "CA_1"]
            logging.info(f"seperated data from 'CA_1' store from 'sell_prices_df'")

            main_df = sales_train_validation_melted_df_with_calendar.merge(sale_prices_df_CA_1, how="left", on=["store_id", "item_id", "wm_yr_wk"])
            logging.info(f"merged 'sales_train_validation_melted_df_with_calendar_for_CA_1_store' and 'sale_prices_df_CA_1' dataframes")

            main_df = main_df.replace(["NAN", "Nan", "nan", "Na", "na", "NA"], np.nan)
            logging.info(f"replaced nan value with np.nan")

            main_df["rolling_1"] = (
                main_df.groupby("id")["sales"]
                .transform(lambda x : x.rolling(1).mean())
            )
            logging.info(f"created 'rolling_1' column")

            main_df["rolling_7"] = (
                main_df.groupby("id")["sales"]
                .transform(lambda x : x.rolling(7).mean())
            )
            logging.info(f"created 'rolling_7' column")

            main_df["rolling_28"] = (
                main_df.groupby("id")["sales"]
                .transform(lambda x : x.rolling(28).mean())
            )
            logging.info(f"created 'rolling_28' column")

            main_df["date"] = pd.to_datetime(main_df["date"])
            

            actual_df = main_df.set_index("date")
            logging.info(f"set 'date' column as index")

            actual_df = actual_df.drop(columns=["id", "d", "wm_yr_wk", "snap_TX", "snap_WI"])

            actual_df.to_csv(self.feature_store_file_name)
            logging.info(f"exported feature set")

            train_set = actual_df[actual_df.index <= training_pipeline.DATA_SET_SPLITTER]
            logging.info(f"exported train set")

            test_set = actual_df[actual_df.index >= training_pipeline.DATA_SET_SPLITTER]

            logging.info(f"exported test set")

            train_set = train_set.reset_index()
            test_set = test_set.reset_index()

            train_set.to_csv(self.train_set_file_name, index=False)
            test_set.to_csv(self.test_set_file_name, index=False)

            logging.info(f"exported data ingestion artifacts")

            return DataIngestionArtifact(train_file_path=self.train_set_file_name, test_file_path=self.test_set_file_name, feature_store_path=self.feature_store_file_name)
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)