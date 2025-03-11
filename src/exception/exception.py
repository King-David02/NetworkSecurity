import sys
from src.logging.logger import logging

class CustomException(Exception):
    def __init__(self, error, error_details:tuple):
        super().__init__(error)
        self.error = error
        _,_,tb = error_details
        self.filename = tb.tb_frame.f_code.co_filename
        self.line_number = tb.tb_lineno

        def __str__(self):
            return f'The "Error" {self.error}, occurred in "File" {self.filename}, at line {self.line_number}'
        

# Example Usage
#try:
    logging.info("correction")
#  1 / 0 
#except Exception as e:
#    raise CustomException(e, sys.exc_info())
