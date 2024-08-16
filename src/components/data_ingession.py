import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    data_path = os.path.join('artifacts','student_data.csv')
    train_data_path = os.path.join('artifacts','train_student_data.csv')
    test_data_path = os.path.join('artifacts','test_student_data.csv')
class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()
    def initilize_data_ingestion(self):
        try:
            logging.info("Data ingestion initilised")
            df = pd.read_csv('./notebook/data/student_data.csv')
            logging.info("completed reading of raw data")
            os.makedirs(os.path.dirname(self.config.data_path),exist_ok=True)
            df.to_csv(self.config.data_path,index=False)
            train_df , test_df = train_test_split(df,test_size =0.2 , random_state = 42)
            logging.info("completed splitting : tarin and test")
            train_df.to_csv(self.config.train_data_path)
            test_df.to_csv(self.config.test_data_path)
            logging.info("created train and test data")
            return(
                self.config.train_data_path,
                self.config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__ =="__main__" :
    obj = DataIngestion()
    obj.initilize_data_ingestion()
        