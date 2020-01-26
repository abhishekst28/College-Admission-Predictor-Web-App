# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 15:48:05 2020

@author: toshn
"""

import numpy as np
from flask import Flask ,request,jsonify,render_template
import logging
from logging.handlers import RotatingFileHandler
import pickle


app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    for x in  request.form.values():
        app.logger.info(x)
    int_features = [float(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)
    
    output=abs(round(prediction[0],2))
    
    return render_template('index.html',chances_text='Your percentage of getting in an admission should be {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)