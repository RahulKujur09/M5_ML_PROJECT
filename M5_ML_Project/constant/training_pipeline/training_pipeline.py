import os, sys
from datetime import datetime
'''
COMMON CONSTANTS
'''
REQUIREMENTS_FILE_PATH : str = os.path.join(os.getcwd(),"requirements.txt")

LOG_DIR : str = "logs"
LOG_FILE_NAME : str = format(datetime.now(), "%d_%m_%y_%H_%M_%S") + ".log"

ARTIFACT_DIR_NAME : str = "ARTIFACTS"

DATA_SET_SPLITTER : str = "2016-01-01"

CALENDAR_CSV_FILE_PATH : str = os.path.join(os.getcwd(), "data", "calendar.csv")
SALES_TRAIN_VALIDATION_PATH : str = os.path.join(os.getcwd(), "data", "sales_train_validation.csv")
SELL_PRICES_PATH : str = os.path.join(os.getcwd(), "data", "sell_prices.csv")



'''
DATA INGESTION CONSTANT starts with DATA_INGESTION_VAR_NAME
'''

DATA_INGESTION_DIR_NAME : str = "data_ingestion"
DATA_INGESTION_FEATURES_DIR_NAME : str = "feature_store"
DATA_INGESTION_FEATURE_FILE_NAME : str = "features.csv"
DATA_INGESTION_DATA_SETS_DIR : str = "data_set"
DATA_INGESTION_TRAIN_SET_FILE_NAME : str = "train.csv"
DATA_INGESTION_TEST_SET_FILE_NAME : str = "test.csv"
