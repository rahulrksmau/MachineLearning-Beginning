#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 10:05:36 2017

@author: standarduser
"""
from scipy.spatial import distance
def euc(a,b):
    return distance.euclidean(a,b)


class Define_fn():
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self,X_test):
        predict = []
        for row in X_test:
            label = self.closest(row)
            predict.append(label)
        
        return predict
    
    def closest(self,row):
        best_dist = euc(row, X_train[0])
        best_index= 0
        for i in xrange(1,len(self.X_train)):
            dist = euc(row , self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index= i
        return self.y_train[best_index]


from sklearn import datasets
from sklearn import model_selection

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train,X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.7)

my_clf = Define_fn()

my_clf.fit(X_train, y_train)

predictions = my_clf.predict(X_test)

from sklearn.metrics import accuracy_score

print accuracy_score(y_test, predictions)