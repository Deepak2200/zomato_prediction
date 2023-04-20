import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from src.utils import save_object


@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")

class DataTranformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationconfig()

    def get_data_transformation(self):
        try:
            logging.info("data tranformation strated")
            #by train model
            cat_col=['Weather_conditions', 'Road_traffic_density', 'Type_of_order',
                'Type_of_vehicle','City'],
                
            num_col=['Delivery_person_Age', 'Delivery_person_Ratings', 'Restaurant_latitude',
                'Restaurant_longitude', 'Delivery_location_latitude',
                'Delivery_location_longitude', 'Vehicle_condition',
                'multiple_deliveries', 'order_day', 'order_month', 'order_hours','Festival'
                'order_min', 'picked_hours', 'picked_min'],
                



            #by train model
            wheather=['Sandstorms','Fog', 'Stormy', 'Windy', 'Cloudy', 'Sunny']
            road_jam=['Jam', 'High', 'Medium', 'Low']
            food_order=['Drinks', 'Buffet','Snack', 'Meal']
            vehical_cat=["bicycle","electric_scooter","scooter","motorcycle"]
            festival=["No","Yes"]
            city_cat=['Metropolitian', 'Urban', 'Semi-Urban']

            logging.info("pipeline initiated")
            # copy pipeline
            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

                ]

            )

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[wheather,road_jam,food_order,vehical_cat,festival,city_cat])),
                ('scaler',StandardScaler())
                ]

            )

            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,num_col),
            ('cat_pipeline',cat_pipeline,cat_col)
            ])

            return preprocessor

            logging.info("pipline completed")




        except Exception as e:
            logging.info("Error rises in Data Transformation")
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            logging.info("start the reading data train and test data")
            #reading the train and test data
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("read train and test data completed")
            logging.info(f"Train dataframe head: \n{train_df.head().to_string()}")
            logging.info(f"test dataframe head: \n{test_df.head().to_string()}")

            logging.info("obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformation()

            
            traget_colums="Time_taken (min)"
            drop_columns=[traget_colums,"ID", "Delivery_person_ID"]

            input_feature_train_df=train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[traget_colums]

            input_feature_test_df=train_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=train_df[traget_colums]

            logging.info("Applying preprocessure object on train  test data")

            #transformating using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Appling preprocessing object on training and testing data set")

            #for easy to read
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            #we need to save pkl.file all the time 

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("preprocessor pickel  filr saved")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info("excpetion occure in the initiate data transform")
            raise CustomException(e,sys)




            






        
        