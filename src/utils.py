import pandas as pd
import os
import sys
import pickle
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