from src.logger import logging
from src.exception import CustomException
import sys , os 
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor ,GradientBoostingRegressor,RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from src.utils import save_object ,evaluate_models
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

@dataclass
class DataTrainingConfig:
    model_path = os.path.join('artifacts','model.pkl')
class DataTraining:
    def __init__(self):
        self.path = DataTrainingConfig()
    def model_training(self , train_arr,test_arr):
        try:
            logging.info("inside model trainer")
            x_train = train_arr[:,:-1]
            y_train =train_arr[:,-1]
            x_test = test_arr[:,:-1]
            y_test = test_arr[:,-1]
            models ={

            "linear regression" :LinearRegression(),
            "ada boost regressor" : AdaBoostRegressor(),
            "gradient boost regressor" :GradientBoostingRegressor(),
            "random forest regressor" :RandomForestRegressor(),
            "decision tree regressor" : DecisionTreeRegressor(),
            "knn" :     KNeighborsRegressor(),
            "xgboost regressor" :XGBRegressor()
            }

            model_report = evaluate_models(models ,x_train,y_train,x_test,y_test)
            bes_model = list(sorted(model_report.items() , key = lambda x: x[1]))[-1]
            best_model_name ,best_model_score = bes_model[0] , bes_model[1]
            logging.info(f"best model is {best_model_name} with score {best_model_score}")
            best_model_obj = models[best_model_name]
            save_object(self.path.model_path,best_model_obj)
            logging.info(f"{best_model_name} saved succesfully at {self.path.model_path}")
            predicted=best_model_obj.predict(x_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square
        except Exception as e:
            raise CustomException(e,sys)
            
        
