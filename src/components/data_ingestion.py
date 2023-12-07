import os
import sys

from src.exceptions import CustomException 
from src.logger import logging 
from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation

from dataclasses import dataclass

from sklearn.model_selection import train_test_split
import pandas as pd


@dataclass
class DataIngestionConfig:
    train_data_path : str=os.path.join('artifacts',"train.csv")
    test_data_path : str=os.path.join('artifacts',"test.csv")
    raw_data_path : str=os.path.join('artifacts',"raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Started the Data Ingestion Process")

        try:

            df = pd.read_csv('notebook\data\Telco-Customer-Churn.csv')
            logging.info("Read the dataset")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) 
            df.to_csv(self.ingestion_config.raw_data_path,index=False)

            logging.info("Train Test Split initiated")

            train_set , test_set = train_test_split(df,test_size=0.2,random_state=42)

            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False)

            logging.info("Data Ingestion Completed")

            return(

                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data_path , test_data_path = data_ingestion.initiate_data_ingestion()

    print(train_data_path)
    print(test_data_path)

    data_transformation = DataTransformation()

    train_arr,test_arr,processor_path = data_transformation.initiate_data_transformation(train_data_path,test_data_path)

    print(train_arr)
    print(test_arr)
    print(processor_path)
    