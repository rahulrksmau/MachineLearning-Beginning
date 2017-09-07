#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 16:35:46 2017

@author: standarduser
"""
import numpy as np
X =  np.array([15, 12, 8, 8, 7, 7, 7, 6, 5, 3])
Y = np.array([10,25,17,11,13,17,20,13,9,15])

n = len(X)
up = (n*sum(X*Y)-sum(X)*sum(Y))
down = ((n*sum(X**2)-(sum(X))**2)*(n*sum(Y**2)-(sum(Y))**2))**0.5

A_const = float(sum(Y)*sum(X**2) - sum(X)*sum(X*Y))/(n*sum(X**2)-sum(X)**2)
B_slope = float(n*sum(X*Y) - sum(X)*sum(Y))/(n*sum(X**2) - sum(X)**2)

Karl_Pearson_coeff = float(up/down)
slop_of_reregression = B_slope

print "%.3f"%Karl_Pearson_coeff
print "%.3f"%slop_of_reregression
