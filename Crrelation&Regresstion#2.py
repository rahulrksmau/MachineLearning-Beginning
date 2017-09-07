#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 17:37:30 2017

@author: standarduser
"""

X =  [[15],[ 12], [8], [8],[ 7],[ 7],[ 7],[ 6], [5], [3]]
Y = [[10],[25],[17],[11],[13],[17],[20],[13],[9],[15]]

n = len(X)
'''
from sklearn import svm
clf = svm.SVC()

from sklearn import tree
clf = tree.DecisionTreeClassifier()

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
clf.fit(X,Y)
print round(clf.predict(10),1)
'''


from sklearn import linear_model
import matplotlib.pyplot as plt

regr = linear_model.LinearRegression()
'''regr.fit(X, Y)
print round(regr.predict(10),1)

print regr.intercept_
print regr.coef_
#plt.scatter(X,Y)'''

d = [1,2,3,4]
plt.plot(d,[4.54,5.53,6.56,5.54], color='r')
plt.plot(d,[30.54,27.53,24.42,20.11], color='b')
plt.show()

S = [[4.54], [5.53], [6.56], [5.54]]
H = [[30.54], [27.53], [24.42], [20.11]]
d = [[1],[2],[3],[4]]
regr.fit(d,S)
print regr.predict([5])

regr.fit(d,H)
print regr.predict([5])