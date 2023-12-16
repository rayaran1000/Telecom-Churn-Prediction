import os
import sys
import dill

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

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
def evaluate_model(X_train, y_train, X_test , y_test, models, params):
    try:
        report = {}

        stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        for model_name, model in models.items():
            param = params[model_name]

            gs = GridSearchCV(model, param, cv=stratified_kfold)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_

            # Perform Stratified K-Fold cross-validation
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=stratified_kfold, scoring='roc_auc')

            # Train the best model on the training dataset
            best_model.fit(X_train, y_train)

            # Make predictions on the entire dataset
            y_pred = best_model.predict(X_test)

            # Calculate the ROC AUC score on the entire dataset
            model_score = roc_auc_score(y_test, y_pred)

            report[model_name] = {
               # 'best_params': gs.best_params_,
               # 'cv_scores': cv_scores,
                'overall_score': model_score
            }

        return report

# Example usage:
# report = evaluate_model(X, y, models, param)
# print(report)
   
    except Exception as e:
        raise CustomException(e,sys)
    
    