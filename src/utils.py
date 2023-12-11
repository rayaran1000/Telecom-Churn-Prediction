import os
import sys
import dill

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

from src.exceptions import CustomException

def save_object(file_path,obj): # Used to save an object, used by us to save pickle files
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj, file_obj)# Dill used to create the pickle files

    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):# Function used to load the pickle files

    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)
    
#Function created to evaluate the models and create a dictionary report consisting of the error values and the model names for both train and test datasets
def evaluate_model(X_train,y_train,X_test,y_test,models,param):
    try:
        report ={}

        for i in range(len(list(models))):
            model=list(models.values())[i]
            para = param[list(models.keys())[i]] #Extracting the parameters for each model

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_) # The best parameters we get are used for the respective model
            model.fit(X_train,y_train)  # We are training the model here with the best parameters.
            

            y_train_pred = model.predict(X_train) # Predictions from the model
            y_test_pred = model.predict(X_test)

            train_model_score = roc_auc_score(y_train,y_train_pred) # R2 scores of the training and test datasets for the models
            test_model_score = roc_auc_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = (test_model_score,gs.best_params_) # Adding all the reports for each individual model in the report dictionary

        return report
    
    except Exception as e:
        raise CustomException(e,sys)
    
    