import sys
import os
import pandas as pd
from src.exceptions import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl") 
            preprocessor_path=os.path.join('artifacts','processor.pkl') #Will handle the preprocessing part
            print("Before Loading")
            model=load_object(file_path=model_path) #Loads the Model from the Pickle file
            preprocessor=load_object(file_path=preprocessor_path) #Loads the Preprocessor from Pickle file
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData: # Class responsible for mapping all the inputs that we are getting in the HTML webpage with the backend
    def __init__(self,        
        gender: str,
        senior_citizen:int,
        partner:str,
        dependents:str,
        tenure_in_months:int,
        phone_service:str,
        multiple_lines:str,
        internet_service:str,
        online_security:str,
        online_backup:str,
        device_security:str,
        tech_support:str,
        streaming_tv:str,
        streaming_movies:str,
        contract:str,
        paperless_billing:str,
        payment_method:str,
        monthly_charges:float,
        total_charges:str):

#Assigning these values(coming from web application)
        self.gender = gender

        self.senior_citizen = senior_citizen

        self.partner = partner

        self.dependents = dependents

        self.tenure_in_months = tenure_in_months

        self.phone_service = phone_service

        self.multiple_lines = multiple_lines

        self.internet_service = internet_service

        self.online_security = online_security

        self.online_backup = online_backup

        self.device_security = device_security

        self.tech_support = tech_support

        self.streaming_tv = streaming_tv

        self.streaming_movies = streaming_movies

        self.contract = contract

        self.paperless_billing = paperless_billing

        self.payment_method = payment_method

        self.monthly_charges = monthly_charges

        self.total_charges = total_charges

    def get_data_as_data_frame(self): #Returns all our input data as dataframe, because we train our models using dataframes
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "SeniorCitizen": [self.senior_citizen],
                "Partner": [self.partner],
                "Dependents": [self.dependents],
                "tenure": [self.tenure_in_months],
                "PhoneService": [self.phone_service],
                "MultipleLines": [self.multiple_lines],
                "InternetService": [self.internet_service],
                "OnlineSecurity": [self.online_security],
                "OnlineBackup": [self.online_backup],
                "DeviceProtection": [self.device_security],
                "TechSupport": [self.tech_support],
                "StreamingTV": [self.streaming_tv],
                "StreamingMovies": [self.streaming_movies],
                "Contract": [self.contract],
                "PaperlessBilling": [self.paperless_billing],
                "PaymentMethod": [self.payment_method],
                "MonthlyCharges": [self.monthly_charges],
                "TotalCharges": [self.total_charges]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)