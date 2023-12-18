import os
import sys

#Libraries for logging and Custom Exception
from src.exceptions import CustomException 
from src.logger import logging 

#Libraries for data ingestion
from src.components.data_ingestion import DataIngestionConfig
from src.components.data_ingestion import DataIngestion

#Libraries for data transformation
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

#Libraries for model trainer
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    
    try:

        data_ingestion = DataIngestion()
        
        raw_data_path,test_data_path,train_data_path = data_ingestion.initiate_data_ingestion()
        
        data_transformation = DataTransformation()

        train_arr,test_arr,processor_path = data_transformation.initiate_data_transformation(train_data_path,test_data_path)

        model_trainer = ModelTrainer()

        best_model_name , roc_score = model_trainer.initiate_model_trainer(train_arr,test_arr)

    except Exception as e:
        raise CustomException(e,sys)
    
    

