#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 17:36:52 2017

@author: standarduser
"""
#working on pipline
# dividing data into 2 category

# import datasets
from sklearn import datasets

# predefine dataset available in sklearn 
iris = datasets.load_iris()

# can use as f(x) = y
# iris have attributes = [data,feature_name,target,target_name]
X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split

# test_size parameter between (0-1) percentage of seperation 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5)


# prediction on the basis of decision Tree
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(X_train,y_train)
#predictions 
prediction = clf.predict(X_test)
print prediction
# checking accuracy of prediction IN THIS CASE >> 0.946666666667
from sklearn.metrics import accuracy_score
print "prediction on basis of Decision Tree ",
print accuracy_score(y_test, prediction)



# predictions on the besis of KNeighbor classifier 
from sklearn.neighbors import KNeighborsClassifier
clfr = KNeighborsClassifier()
clfr.fit(X_train, y_train)
#predict
predictions = clfr.predict(X_test)
print predictions
# test 
print "prediction on basis of K-Neighbors Classifier ",
print accuracy_score(y_test, predictions)
