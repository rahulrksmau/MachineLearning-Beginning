#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 16:44:19 2017

@author: standarduser
"""
from sklearn import random_projection, datasets, svm, metrics, model_selection
import numpy as np

rng = np.random.RandomState(0)
X  = rng.rand(10,2000)
X  = np.array(X, dtype='float32')

tranform = random_projection.GaussianRandomProjection()
X_new = tranform.fit_transform(X)

digits = datasets.load_digits()
#digits = digits.images

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
#plt.imshow(digits[-2], cmap=plt.cm.gray_r)
#plt.show()


# training and prediction for digit recognization

training_data = list(zip(digits.images, digits.target))
'''for index, (image,label) in enumerate(training_data[:3]):
    plt.subplot(2,3, index+1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(' % i'% label)
'''
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

X_train, X_test, y_train, y_test = model_selection.train_test_split(data, digits.target, test_size=0.5)
#creating classifier : support vector classifier

clf = svm.SVC(gamma=0.001)
clf.fit(X_train,y_train)


# test data prediction 
predictions  = clf.predict(X_test)


print "%  {0}".format(metrics.accuracy_score(y_test,predictions))

print ("Classification predicted  %s :\n%s\n" 
       %(clf, metrics.classification_report(y_test, predictions)))
print ("Confusion matrix :\n%s" %metrics.confusion_matrix(y_test, predictions))
'''
prediction_result = list(zip(X_test, predictions))
for index, (image,prediction) in enumerate(prediction_result[:3]):
    plt.subplot(2,3, index+4)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(' % i'% prediction)
 
plt.show()
'''
