import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.exceptions import CustomException
from src.logger import logging
from src.utils import save_object

#Pipeline Creating Libraries
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

#Pre Processing libraries
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

#No missing values pipeline created because in EDA, we found there are no missing values present.

@dataclass
class DataTransformationConfig:
    processor_obj_file_path : str = os.path.join('artifacts',"processor.pkl")

class DataTransformation:

    def __init__(self):
        self.data_processor_config = DataTransformationConfig()

    def get_transformer_object(self,train_path):

        try:
            df = pd.read_csv(train_path)
            categorical_columns = list(df.select_dtypes(include=['object']).columns)
            numerical_columns = list(df.select_dtypes(exclude=['object']).columns)
            print(categorical_columns)
            print(numerical_columns)
            #numerical_columns = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
            #categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']
            

           # self.numerical_columns = numerical_columns
           # self.categorical_columns = categorical_columns

            #Numerical column pipeline
            num_pipeline = Pipeline (
                steps=[

                    ("scaler",StandardScaler(with_mean=False))

                ]
            )

            logging.info("Numerical Column Processing completed")

            #Categorical column pipeline
            cat_pipeline = Pipeline(
                steps=[

                    ("one hot encoder",OneHotEncoder()),

                    ("scaler",StandardScaler(with_mean=False))

                ]
            )
           
            logging.info("Categorical Column Preprocessing completed")

            preprocessor_pipeline = ColumnTransformer(

                [

                ("numerical column preprocessing",num_pipeline,numerical_columns),

                ("categorical column preprocessing",cat_pipeline,categorical_columns)

                ]

            )

            return preprocessor_pipeline
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:

            train_data = pd.read_csv(train_path)           
            test_data = pd.read_csv(test_path)

            logging.info("Training and Testing Data read successfully")

            preprocessor_obj = self.get_transformer_object(train_path)

            target_column_name = "Churn"

            input_feature_train_df=train_data.drop(columns=[target_column_name],axis=1) 
            target_feature_train_df=train_data[target_column_name]

            input_feature_test_df=test_data.drop(columns=[target_column_name],axis=1) 
            target_feature_test_df=test_data[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            print(input_feature_train_df)
            print(input_feature_train_df.shape)

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df) 
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            logging.info("Applied Preprocessing transformations on Train and Test datasets")

            preprocessed_train_arr = np.c_(
                input_feature_train_arr , np.array(target_feature_train_df)
            )

            preprocessed_test_arr = np.c_(
                input_feature_test_arr, np.array(target_feature_test_df)
            )

            save_object(

                file_path=self.data_processor_config.processor_obj_file_path,
                obj=preprocessor_obj

            )

            return(

                preprocessed_train_arr,
                preprocessed_test_arr,
                self.data_processor_config.processor_obj_file_path

            )
        
        except Exception as e:
            raise CustomException(e,sys)
        



