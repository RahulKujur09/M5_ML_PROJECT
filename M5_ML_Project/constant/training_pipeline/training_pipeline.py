import os, sys
from datetime import datetime
'''
COMMON CONSTANTS
'''
REQUIREMENTS_FILE_PATH : str = os.path.join(os.getcwd(),"requirements.txt")

LOG_DIR : str = "logs"
LOG_FILE_NAME : str = format(datetime.now(), "%d_%m_%y_%H_%M_%S") + ".log"

ARTIFACT_DIR_NAME : str = "ARTIFACTS"

ARTIFACT_SUB_FOLDER : str = format(datetime.now(), "%d_%m_%y_%H_%M_%S")

DATA_SET_SPLITTER : str = "2016-01-01"

CALENDAR_CSV_FILE_PATH : str = os.path.join(os.getcwd(), "data", "calendar.csv")
SALES_TRAIN_VALIDATION_PATH : str = os.path.join(os.getcwd(), "data", "sales_train_validation.csv")
SELL_PRICES_PATH : str = os.path.join(os.getcwd(), "data", "sell_prices.csv")

SCHEMA_FILE_PATH : str = "D:/M5_ML_Project/schema/schame.yaml"



'''
DATA INGESTION CONSTANT starts with DATA_INGESTION_VAR_NAME
'''

DATA_INGESTION_DIR_NAME : str = "data_ingestion"
DATA_INGESTION_FEATURES_DIR_NAME : str = "feature_store"
DATA_INGESTION_FEATURE_FILE_NAME : str = "features.csv"
DATA_INGESTION_DATA_SETS_DIR : str = "data_set"
DATA_INGESTION_TRAIN_SET_FILE_NAME : str = "train.csv"
DATA_INGESTION_TEST_SET_FILE_NAME : str = "test.csv"

'''
DATA VALIDATION CONSTANTS starts with DATA_VALIDATION_VAR_NAME
'''
DATA_VALIDATION_DIR_NAME : str = "data_validation"

DATA_VALIDATION_DRIFT_REPORT_DIR : str = "drift_report"

DATA_VALIDATION_DRIFT_FILE_NAME : str = "drift_report.yaml"

DATA_VALIDATION_VALID_DATA_DIR : str = "valid_data"

DATA_VALIDATION_VALID_TRAIN_FILE_NAME : str = "valid_train.csv"
DATA_VALIDATION_VALID_TEST_FILE_NAME : str = "valid_test.csv"

DATA_VALIDATION_INVALID_DATA_DIR : str = "invalid_data"

DATA_VALIDATION_INVALID_TRAIN_FILE_NAME : str = "invalid_train.csv"
DATA_VALIDATION_INVALID_TEST_FILE_NAME : str = "invalid_test.csv"

'''
DATA TRANSFORMATION CONSTANTS STARTS WITH DATA_TRANSFORMATION_VAR_NAME
'''

DATA_TRANSFORMATION_DIR_NAME : str = "data_transformation"

DATA_TRANSFORMATION_TRAIN_FILE_NAME : str = "final_train_set.csv"
DATA_TRANSFORMATION_TEST_FILE_NAME : str = "final_test_set.csv"


'''
MODEL TRAINER CONSTANTS STARTS WITH MODEL_TRAINER_VAR_NAME
'''

# MODEL_TRAINER_DIR_NAME : str = "models"
# MODEL_TRAINER_MODEL_FILE_NAME : str = "final_model.pkl"
# MODEL_TRAINER_REPORT_FILE_NAME : str = "report.yaml"