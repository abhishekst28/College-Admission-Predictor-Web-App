# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 14:41:03 2020

@author: toshn
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

df=pd.read_csv('Admission_Prediction.csv')

#dataset

df=df.drop(columns=['Serial'])
X=df.iloc[:,:7]   

y=df.iloc[:,-1]

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()        

regressor.fit(X,y)
pickle.dump(regressor,open('model.pkl','wb'))

model= pickle.load(open('model.pkl','rb'))

print(model.predict([[320,118,3,4,3,8,1]]))
