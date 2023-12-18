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
            senior_citizen = int(request.form.get('senior_citizen')),
            partner = request.form.get('partner'),
            dependents = request.form.get('dependents'),
            tenure_in_months = int(request.form.get('tenure_in_months')),
            phone_service = request.form.get('phone_service'),
            multiple_lines = request.form.get('multiple_lines'),
            internet_service = request.form.get('internet_service'),
            online_security = request.form.get('online_security'),
            online_backup = request.form.get('online_backup'),
            device_security = request.form.get('device_security'),
            tech_support = request.form.get('tech_support'),
            streaming_tv = request.form.get('streaming_tv'),
            streaming_movies = request.form.get('streaming_movies'),
            contract = request.form.get('contract'),
            paperless_billing = request.form.get('paperless_billing'),
            payment_method = request.form.get('payment_method'),
            monthly_charges = float(request.form.get('monthly_charges')),
            total_charges = request.form.get('total_charges')
        )
        pred_df=data.get_data_as_data_frame() # We are getting the dataframe here
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results = predict_pipeline.predict(pred_df) # Here we are sending the dataframe we created in earlier step for preprocessing and model prediction
        return render_template('home.html',results=results[0]) #Since results will be in list format
    
if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True) # Maps with 127.0.0.1