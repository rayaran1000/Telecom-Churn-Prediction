import os
import sys

from src.exceptions import CustomException 
from src.logger import logging 
from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
from sklearn.model_selection import StratifiedKFold

from dataclasses import dataclass

from sklearn.model_selection import train_test_split
import pandas as pd


@dataclass
class DataIngestionConfig:
    raw_data_path : str=os.path.join('artifacts',"raw.csv")
    test_data_path : str=os.path.join('artifacts',"test.csv")
    train_data_path : str=os.path.join('artifacts',"train.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Started the Data Ingestion Process")

        try:

            df = pd.read_csv('notebook\data\Telco-Customer-Churn.csv')
            X = df.drop('Churn',axis=1)
            y = df['Churn']
            logging.info("Read the dataset")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True) 
            df.to_csv(self.ingestion_config.raw_data_path,index=False)

#Applying train test splits to get training and testing data(used stratify because of imbalance in dataset)
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

            train_data = pd.concat([X_train,y_train],axis=1)
            test_data = pd.concat([X_test,y_test],axis=1)

#Saving the training and testing data in seperate csv files
            train_data.to_csv(self.ingestion_config.train_data_path,index=False)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False)

            logging.info("Finished the Data Ingestion Process")

            return(

                self.ingestion_config.raw_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.train_data_path

            )
        
        except Exception as e:
            raise CustomException(e,sys)
        

    
