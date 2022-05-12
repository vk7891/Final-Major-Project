from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
model = load_model("houseprice.h5")

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():

    Bedrooms = request.form["Bedrooms"]
    Bathrooms = request.form["Bathrooms"]
    Sqft_Living = request.form["Sqft_Living"]
    Sqft_Lot = request.form["Sqft_Lot"]
    Floors = request.form["Floors"]
    Waterfront = request.form["Waterfront"]
    View = request.form["View"]
    Condition = request.form["Condition"]
    Sqft_Above = request.form["Sqft_Above"]
    Sqft_Basement = request.form["Sqft_Basement"]
    Yr_Built = request.form["Yr_Built"]
    Yr_Renovated = request.form["Yr_Renovated"]
    Lat = request.form["Lat"]
    Long = request.form["Long"]    
    Sqft_Living15 = request.form["Sqft_Living15"]   
    Sqft_Lot15 = request.form["Sqft_Lot15"]

    df = np.array([Bathrooms,Bedrooms,Sqft_Living,Sqft_Lot,Floors,Waterfront,View,Condition,Sqft_Above,Sqft_Basement,Yr_Built,Yr_Renovated,Lat,Long,Sqft_Living15,Sqft_Lot15]).reshape(-1,16)
    df = df.astype(np.float64)
    #s_scaler = StandardScaler()
    sc=pickle.load(open('scaler.pkl','rb'))
    single_house = sc.transform(df)
    prediction = model.predict(single_house)
  
    return render_template('predict.html', prediction = prediction)

if __name__ == "__main__":
    app.run(debug=True)
