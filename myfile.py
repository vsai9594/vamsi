# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 19:51:58 2019

@author: Visrdhan xcvb
"""






import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 

dataset = pd.read_csv('housing.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

import statsmodels.formula.api as sm
X = np.append(arr=np.ones((489,1)).astype(int),values=X,axis=1)
X_opt = X[:,[0,1,2,3]]
regressor_OLS =sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
