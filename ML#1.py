#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 17:55:30 2017

@author: standarduser
"""
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import quandl
import math
import pickle


df = quandl.get("WIKI/GOOGL")

    
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low'])/df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
forecast_clm = 'Adj. Close'
df.fillna(value = -99999, inplace=True)
forecast_out = int(math.ceil(0.01*len(df)))

df['label']= df[forecast_clm].shift(-forecast_out)

X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)

y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)
clf = pickle.load(open('linearprogramming.pickle', 'rb'))
clf.fit(X_train, y_train)
predict = clf.score(X_test, y_test)
 
forescat_set = clf.predict(X_lately)
df['Forecast']= np.nan

last_date = df.iloc[-1].name
#last_unix = last_date.timestamp()
one_day = 86400
#next_unix = last_unix + one_day

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('DATE')
plt.ylabel('PRICE')
plt.show()
 

# run only once to save classifier
# so that less time required
''' 
with open('linearprogramming.pickle', 'wb') as f:
    pickle.dump(clf,f)'''