import os, sys
from M5_ML_Project.exception.exception import CustomException
from M5_ML_Project.logging.logger import logging
from M5_ML_Project.constant.training_pipeline import training_pipeline

class TrainingPipelineConfig:
    try:
        def __init__(self):
            self.artifact_sub_dir : str = os.path.join(os.getcwd(),training_pipeline.ARTIFACT_DIR_NAME, training_pipeline.ARTIFACT_SUB_FOLDER)

            os.makedirs(self.artifact_sub_dir, exist_ok=True)
            logging.info(f"Made {self.artifact_sub_dir} dir")
    except Exception as e:
        logging.error(e)
        raise CustomException(e, sys)
    
class DataIngestionConfig:
    def __init__(self, training_pipeline_config : TrainingPipelineConfig):

        try:

            self.data_ingestion_dir : str = os.path.join(training_pipeline_config.artifact_sub_dir, training_pipeline.DATA_INGESTION_DIR_NAME)

            os.makedirs(self.data_ingestion_dir, exist_ok=True)

            self.feature_store_dir : str = os.path.join(self.data_ingestion_dir, training_pipeline.DATA_INGESTION_FEATURES_DIR_NAME)

            os.makedirs(self.feature_store_dir, exist_ok=True)

            logging.info(f"Made {self.feature_store_dir} dir")

            self.feature_store_file_name : str = os.path.join(self.feature_store_dir, training_pipeline.DATA_INGESTION_FEATURE_FILE_NAME)

            self.data_set_dir : str = os.path.join(self.data_ingestion_dir, training_pipeline.DATA_INGESTION_DATA_SETS_DIR)

            os.makedirs(self.data_set_dir, exist_ok=True)

            logging.info(f"Made {self.data_set_dir} dir")

            self.train_file_name : str = os.path.join(self.data_set_dir, training_pipeline.DATA_INGESTION_TRAIN_SET_FILE_NAME)

            self.test_file_name : str = os.path.join(self.data_set_dir, training_pipeline.DATA_INGESTION_TEST_SET_FILE_NAME)
        
        except Exception as e:
            logging.error(e, sys)
            raise CustomException(e, sys)

class DataValidationConfig:
    def __init__(self, training_pipeline_config : TrainingPipelineConfig):
        try:
            self.data_validation_dir : str = os.path.join(training_pipeline_config.artifact_sub_dir, training_pipeline.DATA_VALIDATION_DIR_NAME)

            os.makedirs(self.data_validation_dir, exist_ok=True)
            logging.info("made 'data_validation_dir'")

            self.drift_report_dir : str = os.path.join(self.data_validation_dir, training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR)

            os.makedirs(self.drift_report_dir, exist_ok=True)
            logging.info("made 'drift_report_dir'")

            self.drift_report_file_path : str = os.path.join(self.drift_report_dir, training_pipeline.DATA_VALIDATION_DRIFT_FILE_NAME)

            self.valid_data_dir : str = os.path.join(self.data_validation_dir, training_pipeline.DATA_VALIDATION_VALID_DATA_DIR)

            os.makedirs(self.valid_data_dir, exist_ok=True)
            logging.info("made 'valid_data_dir'")

            self.valid_train_file_path : str = os.path.join(self.valid_data_dir, training_pipeline.DATA_VALIDATION_VALID_TRAIN_FILE_NAME)

            self.valid_test_file_path : str = os.path.join(self.valid_data_dir, training_pipeline.DATA_VALIDATION_VALID_TEST_FILE_NAME)

            self.invalid_data_dir : str = os.path.join(self.data_validation_dir, training_pipeline.DATA_VALIDATION_INVALID_DATA_DIR)

            os.makedirs(self.invalid_data_dir, exist_ok=True)
            logging.info("made 'invalid_data_dir'")

            self.invalid_train_file_path : str = os.path.join(self.invalid_data_dir, training_pipeline.DATA_VALIDATION_INVALID_TRAIN_FILE_NAME)

            self.invalid_test_file_path : str = os.path.join(self.invalid_data_dir, training_pipeline.DATA_VALIDATION_INVALID_TEST_FILE_NAME)
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)