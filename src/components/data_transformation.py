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

    def get_transformer_object(self):

        try:

            numerical_columns = ['SeniorCitizen' , 'tenure' , 'MonthlyCharges' , 'TotalCharges']
            categorical_columns = ['gender','Partner','PhoneService','Dependents', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod' ]

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
        
    def initiate_data_transformation(self,train_data_path,test_data_path):

        try:

            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)           

            logging.info("Test and Train Data read successfully")

#Dropping the customer ID column because its unique for each customer
            train_data = train_data.drop('customerID',axis=1)
            test_data = test_data.drop('customerID',axis=1)

#Handling the blank spaces present in Total Charges column and converting it to float
            condition_train = (train_data['TotalCharges'] == ' ')
            condition_test = (test_data['TotalCharges'] == ' ')

            train_data = train_data[~condition_train]
            test_data = test_data[~condition_test]

            train_data['TotalCharges'] = train_data['TotalCharges'].astype(float)
            test_data['TotalCharges'] = test_data['TotalCharges'].astype(float)

            logging.info("Handled the blank spaces in TotalCharges column")

#Calling the pre processing pipeline created earlier 
            preprocessor_obj = self.get_transformer_object()

#Defining our target column
            target_column_name = "Churn"

#Creating the target column dataframe and independent features dataframes.                   

            input_feature_train_df=train_data.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_data[target_column_name]

            input_feature_test_df=test_data.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_data[target_column_name]

            logging.info("Splitted the independent features and target feature")

#Converting 'Yes' and 'No' in target column values to binary values.
            target_feature_train_df = target_feature_train_df.map({'Yes' : 1 , 'No' : 0})
            target_feature_test_df = target_feature_test_df.map({'Yes' : 1 , 'No' : 0})
            logging.info("Converted the target column values to binary")

#Started the preprocessing for independent features
            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")
            
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df) 
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df) 

            logging.info("Applied Preprocessing transformations on Train and Test datasets")

#Concatenating the preprocessed array of independent features and the target binary column
            preprocessed_train_arr = np.c_[
                input_feature_train_arr , np.array(target_feature_train_df)
            ]

            preprocessed_test_arr = np.c_[
                input_feature_test_arr , np.array(target_feature_test_df)
            ]

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
        



