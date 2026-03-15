import os, sys
from M5_ML_Project.exception.exception import CustomException
from M5_ML_Project.logging.logger import logging
from M5_ML_Project.constant.training_pipeline import training_pipeline

class TrainingPipelineConfig:
    try:
        def __init__(self):
            self.artifact_dir : str = os.path.join(os.getcwd(), training_pipeline.ARTIFACT_DIR_NAME)

            os.makedirs(self.artifact_dir, exist_ok=True)
            logging.info(f"Made {self.artifact_dir} dir")
    except Exception as e:
        logging.error(e)
        raise CustomException(e, sys)
    
class DataIngestionConfig:
    def __init__(self, training_pipeline_config : TrainingPipelineConfig):

        try:

            self.feature_store_dir : str = os.path.join(training_pipeline_config.artifact_dir, training_pipeline.DATA_INGESTION_FEATURES_DIR_NAME)

            os.makedirs(self.feature_store_dir, exist_ok=True)

            logging.info(f"Made {self.feature_store_dir} dir")

            self.feature_store_file_name : str = os.path.join(self.feature_store_dir, training_pipeline.DATA_INGESTION_FEATURE_FILE_NAME)

            self.data_set_dir : str = os.path.join(training_pipeline_config.artifact_dir, training_pipeline.DATA_INGESTION_DATA_SETS_DIR)

            os.makedirs(self.data_set_dir, exist_ok=True)

            logging.info(f"Made {self.data_set_dir} dir")

            self.train_file_name : str = os.path.join(self.data_set_dir, training_pipeline.DATA_INGESTION_TRAIN_SET_FILE_NAME)

            self.test_file_name : str = os.path.join(self.data_set_dir, training_pipeline.DATA_INGESTION_TEST_SET_FILE_NAME)
        
        except Exception as e:
            logging.error(e, sys)
            raise CustomException(e, sys)