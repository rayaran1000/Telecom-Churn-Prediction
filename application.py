from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from src.pipelines.prediction_pipeline import CustomData,PredictPipeline

application = Flask(__name__)

app=application

#Route for home page

@app.route('/')
def index():
    return render_template('index.html') # Defining the Index Html Page

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html') # Home.html will contain fields for getting out Input fields
    else: # Else means this is a post request( we will be creating a Custom class in predict pipeline which will be called here)
        data=CustomData( # Here we are getting all the Input values from the webpage
            gender = request.form.get('gender'),
            senior_citizen = int(request.form.get('SeniorCitizen')),
            partner = request.form.get('Partner'),
            dependents = request.form.get('Dependents'),
            tenure_in_months = int(request.form.get('tenure')),
            phone_service = request.form.get('PhoneService'),
            multiple_lines = request.form.get('MultipleLines'),
            internet_service = request.form.get('InternetService'),
            online_security = request.form.get('OnlineSecurity'),
            online_backup = request.form.get('OnlineBackup'),
            device_security = request.form.get('DeviceProtection'),
            tech_support = request.form.get('TechSupport'),
            streaming_tv = request.form.get('StreamingTV'),
            streaming_movies = request.form.get('StreamingMovies'),
            contract = request.form.get('Contract'),
            paperless_billing = request.form.get('PaperlessBilling'),
            payment_method = request.form.get('PaymentMethod'),
            monthly_charges = float(request.form.get('MonthlyCharges')),
            total_charges = float(request.form.get('TotalCharges'))
        )
        pred_df=data.get_data_as_data_frame() # We are getting the dataframe here
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results = predict_pipeline.predict(pred_df) # Here we are sending the dataframe we created in earlier step for preprocessing and model prediction
        return render_template('home.html',results=results[0]) #Since results will be in list format
    
if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True) # Maps with 127.0.0.1