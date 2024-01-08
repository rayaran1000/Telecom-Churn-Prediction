
# Telecom Churn Analysis

![Screenshot 2024-01-08 162047](https://github.com/rayaran1000/Telecom-Churn-Prediction/assets/122597408/d5fa74bc-884e-4e32-98e1-fb00a2aa4990)

### Primary objective
Identify patterns, factors, and trends that contribute to customer churn, which occurs when customers discontinue their engagement with a product, service, or brand. By analyzing this data, we can uncover insights into customer behavior, preferences, and pain points that may lead to churn.

### Secondary objective
Involves the examination of various features, such as customer demographics, usage patterns, satisfaction levels, and interactions, using statistical and machine learning techniques to build predictive models that help businesses proactively address and reduce customer churn.


## Directory Structure 

```plaintext
/project
│   README.md
│   requirements.txt
|   exceptions.py
|   logger.py
|   utils.py
|   application.py
|   setup.py
|   Webpage
└───artifacts
|   └───model.pkl
|   └───processor.pkl
|   └───raw.csv
|   └───test.csv
|   └───train.csv
└───logs
└───notebook
|   └───data
|       └───Telco-Customer-Churn.csv
|       EDA on Churn Prediction   
└───src
|   └───components
|       └───data_ingestion.py
|       └───data_transformation.py
|       └───model_trainer.py
|   └───pipelines
|       └───prediction_pipeline.py
|       └───training_pipeline.py
└───templates
|   └───home.html
|   └───index.html

```
## Installation

For Installing the necessery libraries required 

```bash
  pip install -r requirements.txt
```
    
## Deployment

To deploy this project run

1. To start the training pipeline 

```bash
  python src/pipelines/training_pipeline.py
```

2. Once the model is trained, to run the Flask application

```bash
  python application.py
```

3. Go to 127.0.0.1/predictdata to get the webpage

4. Use Ctrl + C in terminal to stop the server 

## Dataset Used

Kaggle Dataset - Telco Customer Churn Dataset Used

[Dataset Link](https://www.kaggle.com/code/bandiatindra/telecom-churn-prediction/input)

This dataset has 21 columns, namely:

> Customers who left within the last month – the column is called Churn (Target Column)

> Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies

> Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges

> Demographic info about customers – gender, age range, and if they have partners and dependents
## Exploratory Data Analysis Path followed:


> 1. Importing a dataset

> 2. Understanding the big picture

> 3. Preparation / Data Cleaning

> 4. Understanding and exploring Data

> 5. Study of the relationships between variables

> 6. Plotting Data to infer results

> 7. Conclusion


## Model Training and Evaluation

Models used in the pipeline : Logistic Regression , Random Forest , Decision Tree , Gradient Boosting , K Neighbours Classifier , Adaboost Classifier , Support Vector Classifier , Gaussian Naive Bayes , Bagging Classifier

Best Model Selected on Basis on Evaluation Metric : Logistic Regression

Evaluation Metric Used : Area under ROC curve 

Reason behind evaluation Metric used : Since our dataset was imbalanced and we are performing Classification task
## Acknowledgements

I would like to express my gratitude to the following individuals and resources that contributed to the successful completion of this Telecom Churn Prediction project:

- **[Kaggle]**: Special thanks to the Kaggle for providing access to the dataset and valuable insights into the industry challenges.

- **Open Source Libraries**: The project heavily relied on the contributions of the open-source community. A special mention to libraries such as scikit-learn, pandas, and matplotlib, which facilitated data analysis, model development, and visualization.

- **Online Communities**: I am grateful for the support and knowledge shared by the data science and machine learning communities on platforms like Stack Overflow, GitHub, and Reddit.

This project was a collaborative effort, and I am grateful for the support received from all these sources.


