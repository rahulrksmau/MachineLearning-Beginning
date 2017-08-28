#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 12:12:09 2017

@author: standarduser
"""
import numpy as np
import numpy.linalg as np_l
import matplotlib.pyplot as plt
#NumPy's main object is the homogeneous multidimensional array. 
#It is a table of elements (usually numbers), all of the same type, indexed by a tuple of positive integers. 
#In Numpy dimensions are called axes. 
#The number of axes is rank.

arr = np.array([1,2,4])

#ndim (dimension or rank) >>1
#shape(tuple of +ve number) >> (row,col)>> (3,)
#size >> 3


arr1 = np.array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14]])

arr2 = np.arange(15).reshape(3,5)

'''
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14]])
'''

#linspcae divide into given parts
np.linspace(2,20,100)
'''
array([  2.        ,   2.18181818,   2.36363636, ...,  19.63636364,
        19.81818182,  20.        ])
'''
x=np.linspace(0,2*np.pi,100)
'''
array([ 0.        ,  0.06346652,  0.12693304, ...,  6.15625227,
        6.21971879,  6.28318531])
'''

y1=np.sin(np.linspace(0,2*np.pi,100))
'''
array([  0.00000000e+00,   6.34239197e-02,   1.26592454e-01, ...,
        -1.26592454e-01,  -6.34239197e-02,  -2.44929360e-16])

sin_plot = plt.plot(x,y1, label='sin plot')

y2=np.cos(np.linspace(0,2*np.pi,100))

cos_plot = plt.plot(x,y2, label='cos plot')
plt.annotate('sin graph', xy=(x[3],y1[3]), xytext=(3.5,0.65),
             arrowprops = dict(facecolor='blue'))
plt.annotate('cos graph', xy=(x[25],y2[25]), xytext=(3.5,0.45),
             arrowprops = dict(facecolor='green'))
plt.show()
'''
# matrix using numpy

A = np.ones((3,3), dtype = int)

'''
[[1 1 1]
 [1 1 1]
 [1 1 1]]
'''
#scalar multiplication
A *= 12
B = np.ones((3,3))*14
'''print np.dot(A,B)
[[ 504.  504.  504.]   # matrix multiplication
 [ 504.  504.  504.]
 [ 504.  504.  504.]]
'''

arr2 = np.array([[ 7.,  5.,  9.,  3.],
       [ 7.,  2.,  7.,  8.],
       [ 6.,  8.,  3.,  2.]])
#flatting a nd array
arr2.ravel()
arr2 = arr2.transpose()

A = np.array([[1, 3, 7],
       [3, 2, 4],
       [4, 2, 5]])
#element wise multiplication
B = A*A
'''array([[ 1,  9, 49],
       [ 9,  4, 16],
       [16,  4, 25]])
'''

# matrix multiplication
C = np.dot(A,A)
'''array([[38, 23, 54],
       [25, 21, 49],
       [30, 26, 61]])
'''

# identity matrix
I = np.eye(3) # order of 3
 
# inverse of a matrix
A_inv = np_l.inv(A)

#stacking horizontal and virtical

a = np.array([[2,3,1],[5,1,2]])
b = np.array([[5,7,2]])

ab = np.vstack([a,b])
'''
ab = [[2 3 1]
 [5 1 2]
 [5 7 2]]'''

c = np.array([[8],[7]])
ac = np.hstack([a,c])
'''
ac = [[2, 3, 1, 8],
       [5, 1, 2, 7]]
'''