import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from src.components.data_transform import DataTranformation


##initialize the data ingestion congration
@dataclass
class DataIngestionconfig:
    train_data_path:str=os.path.join("artifacts","train.csv")
    test_data_path:str=os.path.join("artifacts","test.csv")
    raw_data_path:str=os.path.join("artifacts","raw.csv")

#create a classs for data ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionconfig()

    
    def initiate_data_ingestion(self):
        logging.info("Data ingestion get start")
        try:
            df=pd.read_csv(os.path.join("notebooks/task_data.csv"))
            logging.info("dataset read from pandas ")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)

            logging.info(f"Train dataframe head: \n{df.head().to_string()}")
            logging.info(f"check for null values: \n{df.isnull().sum().to_string()}")

            ## For the "DATE" column:-

            logging.info("Data cleaing processes start")
            logging.info("we have date columns we need to convert that column into int")
            df["order_day"]=pd.to_datetime(df["Order_Date"],dayfirst=True).dt.day
            df["order_month"]=pd.to_datetime(df["Order_Date"],dayfirst=True).dt.month
            df.drop(labels=["Order_Date"],axis=1,inplace=True)
            logging.info(f"Date column geting convert into int:\n {df.head(2).to_string()}")



            #For the "TIME ORDER" COLUMN :-

            logging.info("time column getting converting start")
            #we convert this into same split form
            df["Time_Orderd"]=df["Time_Orderd"].str.replace(".",":")
            #After replace we need to splite the data hours ans mins
            df["order_hours"]=df["Time_Orderd"].str.split(":").str[0]
            df["order_min"]=df["Time_Orderd"].str.split(":").str[1]
            df["order_hours"]=df["order_hours"].astype(float)
            df["order_min"]=df["order_min"].astype(float)
            logging.info(f"chcek unique values of order_hours:\n {df.order_hours.unique()}")
            logging.info(f"chcek unique values of order_min:\n {df.order_min.unique()}")
            
            #when look into the order_min than we got its not good as we want than we use some auto mation
            for i in df["order_min"]:
                if i>60:
                    p=i-60
                    df["order_min"]=df["order_min"].replace(i,i-p)
            #why 60?-->#bcoz 1 hour have 60 min
            df.drop(labels=["Time_Orderd"],axis=1,inplace=True)
            logging.info(f"time columns get concerted into float \n {df.order_min.unique()}")
            logging.info(f"DataFrame: \n {df.head().to_string()}")

            #FOR THE "PICKED ORDER TIME":-

            logging.info("Time_order_picked geting converted")
            df["Time_Order_picked"]=df["Time_Order_picked"].str.replace(".",":")
            df["picked_hours"]=df["Time_Order_picked"].str.split(":").str[0]
            df["picked_min"]=df["Time_Order_picked"].str.split(":").str[1]
            df["picked_hours"]=df["picked_hours"].astype(float)
            df["picked_min"]=df["picked_min"].astype(float)
            
            #When look into the order_min than we got its not good as we want than we use some auto mation
            for i in df["picked_min"]:
                if i>60:
                    p=i-60
                    df["picked_min"]=df["picked_min"].replace(i,i-p)
            #we need to drop order_picked_time column
            df.drop(labels=["Time_Order_picked"],axis=1,inplace=True)
            logging.info(f"time columns also getting converted into float: \n {df.head().to_string()}")
            df["Festival"]=df["Festival"].map({'No':0, 'Yes':1,"nan":0})

            logging.info("Train Test Split")
            train_set,test_set=train_test_split(df,test_size=0.30,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False)

            logging.info("Ingestion of data id complete")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            logging.info("Error in data ingestion")
            raise CustomException(e,sys)
        

#run data ingestion

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    data_transforms=DataTranformation()
    train_arr,test_arr,_=data_transforms.initiate_data_transformation(train_data,test_data)







            






            
            
            


           

            

            
        