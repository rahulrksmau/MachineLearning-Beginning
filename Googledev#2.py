#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 12:49:18 2017

@author: standarduser
"""

# iris flower dataset already included in sklearn datasets 
from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree

# loading datasets
iris = load_iris()

#print iris.feature_names
#print iris.target_names # print types of flowers 

#for i in xrange(len(iris.target)):
    #print "Example %d : label %s, features %s,"%(i,iris.target[i], iris.data[i])

test_ids = [0,15,75,145]

# preparing training data
train_dataset = np.delete(iris.data, test_ids, axis=0)
train_target  = np.delete(iris.target, test_ids)

# preparing test data
test_target = iris.target[test_ids]
test_dataset= iris.data[test_ids]

# create  new classifier
clf = tree.DecisionTreeClassifier()

# train on training data
clf.fit(train_dataset, train_target)

print clf.predict(test_dataset)

from sklearn.externals.six import StringIO

import pydot

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names=iris.feature_names, 
                     class_names=iris.target_names, filled=True, rounded=True,
                     impurity=False)

graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")