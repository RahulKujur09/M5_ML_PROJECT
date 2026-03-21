import os,sys
import numpy as np
from M5_ML_Project.exception.exception import CustomException
from M5_ML_Project.logging.logger import logging

def reduce_mem(df):
    try:
        for col in df.columns:
            if df[col].dtype == "float64":
                df[col] = df[col].astype(np.float32)
            elif df[col].dtype == "int64":
                df[col] = df[col].astype(np.int32)
            elif df[col].dtype == "object":
                df[col] = df[col].astype("category")    
        return df            
    except Exception as e:
        logging.error(e)
        raise CustomException(e, sys)