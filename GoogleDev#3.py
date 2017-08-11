#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 17:24:01 2017

@author: standarduser
"""

# histogram view
import matplotlib.pyplot as plt
import numpy as np

greyhound = 500
labs = 500

# average height of labrador 28"
# average height of greyhound 24"

gr_height = 28 +3*np.random.randn(greyhound)
lb_height = 24 +3*np.random.randn(labs)

plt.hist([gr_height, lb_height], stacked=True, color=['r','b'])
plt.show()