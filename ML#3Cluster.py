#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 17:41:12 2017

@author: standarduser
"""
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans

style.use('ggplot')

X = np.array([[3,2],
              [5,1.5],
              [4,8],
              [6,12],
              [12,45],
              [11,15]])

#plt.scatter(X[:,0], X[:,1], color=['b','r','y'])
plt.show()

clf = KMeans(n_clusters=4)
clf.fit(X)

centroids = clf.cluster_centers_
labels = clf.labels_

colors = ["g.", "r.", "c.", "y."]
for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=15)
plt.scatter(centroids[:, 0], centroids[:,1], marker = "x")
plt.show()