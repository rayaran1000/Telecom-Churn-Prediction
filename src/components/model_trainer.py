import os
import sys
from dataclasses import dataclass

from src.exceptions import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_model

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.metrics import roc_auc_score


@dataclass
class ModelTrainerConfig():
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and test datasets")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = { # Dictionary of Models we will be trying
                "Random Forest" : RandomForestClassifier(),
                "Decision Tree" : DecisionTreeClassifier(),
                "Gradient Boosting" : GradientBoostingClassifier(),
                "Logistic Regression" : LogisticRegression(),
                "K Neighbours Classifier" : KNeighborsClassifier(),
                "Adaboost Classifier" : AdaBoostClassifier(),
                "Support Vector Classifier" : SVC(),
                "Gaussian Naive Bayess" : GaussianNB(),
                "Gradient Boosting" : GradientBoostingClassifier(),
                "Bagging Classifier" : BaggingClassifier()
            }

            params={ # Creating a dictionary with the parameters for each Model
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                     'splitter':['best','random'],
                     'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                     'criterion':['gini', 'log_loss', 'entropy'],
                     'max_features':['sqrt','log2',None],
                     'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                     'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                     'criterion':['squared_error', 'friedman_mse'],
                     'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Logistic Regression":{},
                "K Neighbours Classifier" :{
                        'n_neighbors': [3, 5, 7, 9],
                        'weights': ['uniform', 'distance'],
                        'p' : [1, 2]
                },
                "XGB Classifier":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Adaboost Classifier":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Support Vector Classifier":{
                    'kernel':['linear', 'poly', 'rbf','sigmoid'],
                    'gamma':[0.001,0.01,0.1,1,10],
                    'C':[0.001,0.01,0.1,1,10]
                },
                "Gaussian Naive Bayess":{
                    'var_smoothing':[0.001,0.01,0.1,1,10]
                },
                "Bagging Classifier":{
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                }    
                
            }

            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params) # This function created inside the utils.py

            #Sorting and extracting the best model using the model score
            best_model_score = max(sorted(model_report.values()))

            #To get the best model name
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)] # Nested lists concept used here

            best_model = models[best_model_name]

            #if best_model_score < 0.6: # If all model scores are less than 0.6
            #    raise CustomException( "No best models found") 

            logging.info("Best found model on both training and testing datasets")

            save_object(# Creating the Model.pkl file corresponding to the best model that we will get
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            roc_score = roc_auc_score(y_test,predicted)

            return roc_score

        except Exception as e:
            raise CustomException(e,sys)


