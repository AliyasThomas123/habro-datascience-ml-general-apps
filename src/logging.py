import logging
import os
from datetime import datetime
LOG_FILE_NAME = f"{datetime.now().strftime("%yyyy_%mm_%d_%H_%M_%S")}.log"
LOGS_PATH = os.join(os.getcwd(),"logs",LOG_FILE_NAME)
os.mkdirs(LOGS_PATH,exist_ok = True)
LOG_FILE_PATH = os.join(LOGS_PATH , LOG_FILE_NAME)

logging.basicConfig(

filename=LOG_FILE_PATH ,
format= "{asctime} - {levelname} - {message}" , 
level=logging.INFO

)