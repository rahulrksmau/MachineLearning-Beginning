#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 16:32:08 2017

@author: standarduser
"""

import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

log = np.genfromtxt('traingData.txt', delimiter=',')
charge_time = []
life_time = []

for i in log:
    i = i.tolist()
    temp = []
    temp.append(i[0])
    charge_time.append(temp)
    life_time.append(i[1])
    
plt.scatter(charge_time, life_time, c = 'red')
plt.xlabel('charge time')
plt.ylabel('life time')
plt.title('Laptop Battery life')
plt.show()

'''lin = linear_model.LinearRegression()

lin.fit(charge_time, life_time)
print lin.get_params()

print round( lin.predict(input()), 2)'''