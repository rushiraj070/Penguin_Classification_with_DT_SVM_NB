from flask import Flask, request, app,render_template
from flask import Response
import pickle
import numpy as np
import pandas as pd


application = Flask(__name__)
app=application

dt_model = pickle.load(open("Model/decision_tree.pkl", "rb"))
svm_model = pickle.load(open("Model/support_vector_tuned.pkl", "rb"))
gnb_model = pickle.load(open("Model/gaussian_nb.pkl", "rb"))


## Route for homepage

@app.route('/')
def index():
    return render_template('index.html')

## Route for Single data point prediction
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    result=""

    if request.method=='POST':

        bill_length_mm = float(request.form.get("bill_length_mm"))
        bill_depth_mm = float(request.form.get('bill_depth_mm'))
        flipper_length_mm = float(request.form.get('flipper_length_mm'))
        body_mass_g = float(request.form.get('body_mass_g'))

        predict_dt = dt_model.predict([[bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g]])
        predict_dt = svm_model.predict([[bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g]])
        predict_gnb = gnb_model.predict([[bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g]])
       
        predict = max(predict_dt, predict_dt, predict_gnb)

        if predict[0] == 1 :
            result = 'Yohoo..It is a BOY..!!'
        else:
            result ='Yipee..It is a GIRL..!!'
            
        return render_template('penguin_sex_prediction.html',result=result)

    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")