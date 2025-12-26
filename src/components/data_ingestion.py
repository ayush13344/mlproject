import os
import sys
from src.exception import CustomException   
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()    

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")
        try:
            df=pd.read_csv('notebook\data\studentsPerformance.csv')
            logging.info("Raw data read successfully")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            train_df,test_df=train_test_split(df,test_size=0.2,random_state=42)
            logging.info("Train and test data split successfully")
            train_df.to_csv(self.ingestion_config.train_data_path,index=False)
            test_df.to_csv(self.ingestion_config.test_data_path,index=False)
            logging.info("Train and test data saved successfully")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.error(f"Error occurred during data ingestion: {e}")
                

if __name__=="__main__":
    obj=DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path, test_data_path)

    modeltrainer=ModelTrainer()
    modeltrainer.initiate_model_trainer(train_arr,test_arr)