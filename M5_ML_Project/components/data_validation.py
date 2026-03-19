import os, sys
import pandas as pd
from scipy.stats import ks_2samp
from M5_ML_Project.logging.logger import logging
from M5_ML_Project.exception.exception import CustomException
from M5_ML_Project.entity.config_entity import DataValidationConfig
from M5_ML_Project.entity.artifact_entity import DataIngestionArtifact, DataValidationartifacts
from M5_ML_Project.constant.training_pipeline import training_pipeline
from M5_ML_Project.utils.main_utils import get_schema, save_report

class DataValidation:
    def __init__(self, data_validation_config : DataValidationConfig):
        try:
            self.drift_report_file_path : str = data_validation_config.drift_report_file_path

            self.valid_train_file_path : str = data_validation_config.valid_train_file_path

            self.valid_test_file_path : str = data_validation_config.valid_test_file_path

            self.invalid_train_file_path : str = data_validation_config.invalid_train_file_path

            self.invalid_test_file_path : str = data_validation_config.invalid_test_file_path
        except Exception as e:
            logging.error(e, sys)
            raise CustomException(e, sys)
        
    @staticmethod
    def get_csv(file_path):
        try:
            return pd.read_csv(file_path, low_memory=False)
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)
        
    def initiate_data_validation(self, data_ingestion_artifact : DataIngestionArtifact, threshold : float = 0.05) -> DataValidationartifacts:
        try:
            train_set = DataValidation.get_csv(data_ingestion_artifact.train_file_path)
            logging.info("reterived train set")

            test_set = DataValidation.get_csv(data_ingestion_artifact.test_file_path)
            logging.info("reterived test set")

            schame = set(get_schema(training_pipeline.SCHEMA_FILE_PATH))
            logging.info("retrived schema")

            numerical_columns = train_set.select_dtypes(exclude="str").columns

            report = {}
            status = False

            for col in numerical_columns:
                d1 = train_set[col].dropna()
                d2 = test_set[col].dropna()

                is_dist_found = ks_2samp(d1, d2)
                
                if threshold <= is_dist_found.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status=True

                report[col] = {
                    "p_value" : float(is_dist_found.pvalue),
                    "status" : is_found
                }
                logging.info("drift report generated")

                save_report(content=report, file_path=self.drift_report_file_path)

                if set(train_set.columns.to_list()) == schame:
                    train_set.to_csv(self.valid_train_file_path, index=False)
                else:
                    train_set.to_csv(self.invalid_train_file_path, index=False)

                if set(test_set.columns.to_list()) == schame:
                    test_set.to_csv(self.valid_test_file_path, index=False)
                else:
                    test_set.to_csv(self.invalid_test_file_path, index=False)
                logging.info("exported train and test set to their destination")

            return DataValidationartifacts(drift_report_file_path=self.drift_report_file_path, valid_train_set_file_path=self.valid_train_file_path, valid_test_set_file_path=self.valid_test_file_path, invalid_train_set_file_path=self.invalid_train_file_path, invalid_test_set_file_path=self.invalid_test_file_path)

        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)