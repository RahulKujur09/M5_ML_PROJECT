import os, sys
import yaml
from M5_ML_Project.logging.logger import logging
from M5_ML_Project.exception.exception import CustomException

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