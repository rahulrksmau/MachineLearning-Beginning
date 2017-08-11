#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 09:36:41 2017

@author: standarduser
"""

# import pandas as pd
import datetime
import pandas.io.data as web
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

start = datetime.datetime(2010,1,1)
end   = datetime.datetime(2015,1,1)

df = web.DataReader("XOM", "yahoo", start, end)

print (df.head())

df['Adj. Close'].plot()

plt.show()