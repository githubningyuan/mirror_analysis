#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 09:48:01 2017
@author: ny
"""

import pandas as pd
import numpy as np
import csv
import datetime
import time
#from datetime import datetime, date

#from sklearn.linear_model import LogisticRegression, SGDClassifier
#from sklearn.svm import  SVC
#from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LinearRegression, SGDRegressor
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.svm import  SVR
#from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.cross_validation import train_test_split, cross_val_score
#from sklearn.preprocessing import StandardScaler

    
### INPUT 
X_present = [632, 690, 683, 593, 441, 22.38, 26, 14]
R_target = 329.2
### default constraint
constraint = [1,1,1,1,1,1,0,0]  



### prepare data
dataset = pd.read_csv('1127.csv')
data_ok = dataset[dataset['ok']==1]
data_ok = data_ok[data_ok['speed']<30 ]
X_train, X_test, y_train, y_test = train_test_split(data_ok[data_ok.columns[2:10]],
                     data_ok[data_ok.columns[1]],test_size=0.0,random_state=123)

## train a regressor 
lr = LinearRegression(n_jobs=-1)
lr.fit(X_train, y_train)


## type(X) should be a list (a single vector)
def gradient(regressor, X_):
#    X_ = np.array(X)
    dim_X = X_.shape[1]
    u = np.zeros(dim_X)
    grad = np.zeros(dim_X)
    eps = 1e-5
    f0 = regressor.predict(X_)
    for i in range(dim_X):
        u[i]=1
        f1 = regressor.predict(X_+eps*u)
        grad[i] = (f1-f0)/eps
        u[i]=0
#    for i in np.nonzero(constraint):
#        u[i]=1
#        f1 = regressor.predict(X_+eps*u)
#        grad[i] = (f1-f0)/eps
#        u[i]=0
    return grad
    
def parameter_pred(regressor, X_present, f_target, constraint, steps=100):
    ''' 1. The regressor should be pretrained. 
        2. constraint is a list defining which components of X_present are fixed:
            [1, 1, 0, 1] corresponds to one fixed (3rd)component.
    '''
    X_ = np.array(X_present).reshape(1, -1)
    P_ = np.array(constraint).reshape(1, -1)
    f_present = regressor.predict(X_)[0]
    del_f = (f_target - f_present)/steps
    f_ = f_present
    for i in range(steps):
        partial_grad = gradient(regressor, X_)*P_
        X_ += partial_grad*del_f/np.sum(np.square(partial_grad))
        f_ += del_f
    return f_present, X_, f_

    
R_present, X_output, R_output = parameter_pred(lr, X_present, R_target, constraint)


print(X_present)
print(R_present)
print(X_output[0])
print(R_output)
