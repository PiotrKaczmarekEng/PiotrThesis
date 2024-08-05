# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 09:51:36 2024

@author: spide
"""

# LCOT Ammonia

from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# from 0 to 3000

# available points
x1 = 500
x2 = 3000
y1 = 0.2
y2 = 1.05



X = np.array([x1, x2])
X = X.reshape(-1, 1)
Y = np.array([y1, y2])

model = LinearRegression()
model.fit(X, Y)

dist_array = np.linspace(0, 3000, num=500)
dist_array = dist_array.reshape(-1, 1)
LCOT_array = model.predict(dist_array)


plt.plot(dist_array, LCOT_array, marker='o', linestyle='-', color='b')
# plt.plot(dist_set, lvl30, marker='o', linestyle='-', color='r')
# plt.yscale('log')
# plt.ylim(1, 1000)
# plt.yticks([1, 10, 100, 1000], ['1', '10', '100', '1000'])
plt.xlim(0,3000)
plt.title('LCOT vs Distance')
plt.xlabel('Distance [km]')
plt.ylabel('LCOT [Eur/kg]')
plt.grid(True)
plt.show()


prediction_dist =  np.array([1000])
prediction_dist.reshape(-1, 1)
prediction_LCOT = model.predict(prediction_dist)





#%%

def LCOT_Ammonia(distance):
    m = 0.00034
    c = 0.03
    LCOT_prediction = m*distance+c
    print('Estimated LCOT: ',LCOT_prediction, '[Eur/kg]')
    return LCOT_prediction

LCOT_Ammonia(500)

def LCOT_H2(distance):
    m=1/1600
    c = 0
    LCOT_prediction = m*distance+c
    print('Estimated LCOT: ',LCOT_prediction, '[Eur/kg]')
    return LCOT_prediction
