#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 19:09:26 2021

@author: pranchalsihare
"""

from flask import Flask, request, render_template
from joblib import load

app = Flask("__name__")

encode = load('joblib/c_encoder.save')
scaler = load('joblib/scaler.save')
model = load('joblib/model.save')

def performance(mpg):
    label = ""
    if(mpg<20):
        label = "Low Performance"
    elif(mpg>=20 and mpg<=30):
        label = "Moderate Performance"
    elif(mpg>30):
        label = "High Performance"
    else:
        label = "Unknown"
    return label
    
    
    return label

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/per_predict',methods=['POST'])
def per_predict():
    features = [[x for x in request.form.values()]]
    print(features)
    features[0][-1] = features[0][-1].split()[0] # extracting company name from car name 
    print(features)
    features[0].pop(1) # removing displacement
    print(features)
    print(features[0][-1])
    features[0][-1] = encode.transform([features[0][-1]])
    print(features)
    features = scaler.transform(features)
    print(features)
    mpg = model.predict(features)
    print(mpg)
    per_label = performance(mpg)
    print(per_label)
    return render_template('index.html', prediction_text= 'Car Performance : {label}  \nMiles Per Galon (MPG) : {:.2f}'.format(mpg[0],label=per_label))

if __name__ == "__main__":
    app.run(debug=True)
    
    