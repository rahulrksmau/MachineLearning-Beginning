#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 18:05:33 2017

@author: standarduser
"""
# prediction of fruits on the basis of weight and cover

from sklearn import tree
# 0 - smooth fruit
# 1 - rough fruit
features = [[140, 1], [130, 1], [150, 0], [170, 0]] #[[160,0],[120,1],[180,0],[110,1],[114,1]]

# 0 -  apple, 1- orange
label= [0, 0, 1, 1]  #[1,1,0,1,0]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, label)

# make prediction

print 'orange' if clf.predict([[51.1,0]]) else 'apple'