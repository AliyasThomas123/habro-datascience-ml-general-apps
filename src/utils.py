import pandas as pd
import os
import sys
import pickle
from src.logger import logging
from sklearn.metrics import r2_score
from src.exception import CustomException

def get_columns(df_path):
    df = pd.read_csv(df_path)
    numerical_columns = [column for column in df.columns if df[column].dtype != 'O']
    categorical_columns = [column for column in df.columns if df[column].dtype == 'O']
    return numerical_columns , categorical_columns

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
def evaluate_models(models ,x_train,y_train,x_test,y_test):
    try:
       logging.info("inside evaluate models")
        
       model_report ={}
       for model_name,model_obj in models.items():
           model_obj.fit(x_train,y_train)
           predicted_result =model_obj.predict(x_test)
           r2score =r2_score(predicted_result , y_test)
           model_report[model_name] = r2score
       logging.info(f"Model Report: {model_report} ")
       return model_report    
           


    except Exception as e:
        raise CustomException(e,sys)