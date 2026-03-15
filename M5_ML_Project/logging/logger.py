import os, sys
import logging
from M5_ML_Project.constant.training_pipeline import training_pipeline

LOG_DIR = os.path.join(os.getcwd(),training_pipeline.LOG_DIR)
LOG_FILE_NAME = training_pipeline.LOG_FILE_NAME

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename = os.path.join(LOG_DIR, LOG_FILE_NAME),
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)