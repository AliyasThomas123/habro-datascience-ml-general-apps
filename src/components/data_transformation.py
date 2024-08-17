from src.exception import CustomException
from dataclasses import dataclass
from src.logger import logging
from src.utils import get_columns , save_object
from src.components.data_training import DataTraining
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import sys
import os
@dataclass
class DataTransformationConfig:
    data_processing_obj_path = os.path.join('artifacts','preprocessor.pkl')
class DataTransformation:
    def __init__(self) :
        self.preprocessor = DataTransformationConfig()
    def get_preprocessor_obj(self):
        try:
            logging.info("inside get procrssor obj")
            self.numerical_columns , self.categorical_columns = get_columns(os.path.join("artifacts","student_data.csv")) 
            logging.info("extracted numerical and categorical")
            numerical_pipeline  =Pipeline(

                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scalar",StandardScaler(with_mean=False))
                ]
            )
            categorical_pipeline = Pipeline(

                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("encoder",OneHotEncoder()),
                    ("scalar",StandardScaler(with_mean=False))
                ]
            )
            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline",numerical_pipeline , self.numerical_columns),
                    ("categorical_pipeline",categorical_pipeline,self.categorical_columns)
                ]
            )
            logging.info("created column transformer")
            return   preprocessor
        except Exception as e:
            raise CustomException(e,sys) 
    def initilise_transformation(self,train_df_path , test_df_path):
        try:
            logging.info("initilize tranformation")
            train_df = pd.read_csv(train_df_path)
            test_df =pd.read_csv(test_df_path)
            target_variable='math_score'
            train_feature_df = train_df.drop(columns=[target_variable] , axis=1)
            train_label_df =train_df[target_variable]
            test_feature_df =test_df.drop(target_variable , axis=1)
            test_label_df =test_df[target_variable]
            pre_obj =self.get_preprocessor_obj() 
            logging.info(" got procrssor obj")
            self.numerical_columns.remove(target_variable)
            train_feature_arr = pre_obj.fit_transform(train_feature_df)
            test_feature_arr = pre_obj.fit_transform(test_feature_df)
            train_arr = np.c_[
                train_feature_arr, np.array( train_label_df)
            ]
            test_arr = np.c_[test_feature_arr , np.array( test_label_df)]
            logging.info("createing train test array")

            save_object(

                file_path= self.preprocessor.data_processing_obj_path,
                obj=pre_obj

            )
            logging.info("saved procrssor obj")

            return (
                train_arr,
                test_arr,
                self.preprocessor.data_processing_obj_path,
            )

        except Exception as e:
            raise CustomException(e,sys)

if __name__ =="__main__" :
    obj =DataTransformation()
    model_train_obj = DataTraining()
    train_df_path =os.path.join('artifacts','train_student_data.csv')
    test_df_path = os.path.join('artifacts','test_student_data.csv')
    train_arr , test_arr ,path = obj.initilise_transformation(train_df_path,test_df_path)
    model_train_obj.model_training(train_arr , test_arr)


        