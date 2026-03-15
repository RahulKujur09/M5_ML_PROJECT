import os,sys
# from M5_ML_Project.logging.logger import logging

class CustomException(Exception):
    def __init__(self, error_message, error_details):
        super().__init__(error_message)

        self.error_message = error_message

        _,_,exc_tb = error_details.exc_info()

        self.line_name = exc_tb.tb_lineno

        self.file_name = exc_tb.tb_frame.fo_code.co_filename

    def __str__(self) -> str:
        return f"The error occured in the line number [{self.line_name}] in the file name [{self.file_name}], and the error message is [{self.error_message}]"
    
# if __name__ == "__main__":
#     try:
#         a=1/0
#         print("This will not be printed")
#     except Exception as e:
#         logging.error(e)
#         raise CustomException(e, sys)