import numpy as np
import pandas as pd
import sys
import os
from dataclasses import  dataclass 
from src.exception import CustomException
from src.utils import save_object
from src.logger import logging
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from src.utils import evaluate_model

@dataclass
class ModelTrainconfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainconfig()

    def initate_model_training(self,train_array,test_array):
        try:
            logging.info("Splitting Dependendt and independent variables from train and test")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-6],
                train_array[:,-6],
                test_array[:,:-6],
                test_array[:,-6]
            )
            models={
            'LinearRegression':LinearRegression(),
            'Lasso':Lasso(),
            'Ridge':Ridge(),
            'Elasticnet':ElasticNet()
            }


            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print("\n=========================================================================================")
            logging.info(f"Model Report:{model_report}")

            #to get the best model score from the dictionary
            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                file_path=ModelTrainconfig.trained_model_file_path,
                obj=best_model
            )







        except Exception as e:
            logging.info("Exception occured in model training ")
            raise CustomException(e,sys)
