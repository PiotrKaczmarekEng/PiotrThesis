# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 11:00:48 2024

@author: spide
"""

import numpy as np
import matplotlib.pyplot as plt

# Transport Cost Pipeline computation

#### Input params

dist_set = np.linspace(0, 1400, num=36)
#Temporaryyyyy
dist = 1400

Capacity = 1 # [GW]

HoursYear = 365*24 # [hr]
EnergyProduced = 5160 # [GWh]
CapacityFactor = EnergyProduced/HoursYear # ~60% CF

EnergyTransported = 5160 # [GWh/yr]
CAPEX = 0.12 # MUSD/GW/km
OPEX = 0.8 # % of CAPEX each year
# H2loss = 483/1000 # [ton/km/yr]
r = 0.02 # Interest Rate [%]


def LCOET_func(dist):
    CostPipeYear = np.zeros((4,25))
    
    # Year row
    for i in range(25):
        DF = 1/((1+r)**(i+1))
        CostPipeYear[0][i] = i+1
    # CAPEX row
        CostPipeYear[1][i] = CAPEX*dist*1000000 # [USD]
    # OPEX row
        CostPipeYear[2][i] = (OPEX*0.01*CostPipeYear[1][i]) * DF
        print(DF)
    # Denominator row
        CostPipeYear[3][i] = EnergyTransported / DF
    
    print(CostPipeYear)
    LCOET = (CAPEX + sum(CostPipeYear[2])) / sum(CostPipeYear[3])
    
    return LCOET

LCOET_array = np.zeros(36)
step = 0
for dist in dist_set:
    LCOET_array[step] = LCOET_func(dist)
    step += 1

lvl30 = np.zeros(36)
step = 0
for dist in dist_set:
    lvl30[step] = 30
    step += 1

# #plot LCOET vs km
# LCOET_array.plot(title='')
# plt.xlabel('km')
# plt.ylabel('USD/MWh')


plt.plot(dist_set, LCOET_array, marker='o', linestyle='-', color='b')
plt.plot(dist_set, lvl30, marker='o', linestyle='-', color='r')
plt.yscale('log')
plt.ylim(1, 1000)
plt.yticks([1, 10, 100, 1000], ['1', '10', '100', '1000'])
plt.xlim(0,1500)
plt.title('LCOET vs Distance')
plt.xlabel('Distance [km]')
plt.ylabel('LCOET [USD/MWh]')
plt.grid(True)
plt.show()


plt.plot(dist_set, LCOET_array, marker='o', linestyle='-', color='b')
plt.plot(dist_set, lvl30, marker='o', linestyle='-', color='r')
# plt.yscale('log')
# plt.ylim(1, 1000)
# plt.yticks([1, 10, 100, 1000], ['1', '10', '100', '1000'])
plt.xlim(0,1500)
plt.title('LCOET vs Distance')
plt.xlabel('Distance [km]')
plt.ylabel('LCOET [USD/MWh]')
plt.grid(True)
plt.show()

#%%
#### Computation

# #  TYCHOS NUMBERS (FLOATING)
# # Year
# Year = np.array([2025, 2035, 2050])
# # CAPEX
# CAPEX = np.array([4022700*Capacity_Wind, learning_rate_2035*4022700*Capacity_Wind, learning_rate_2035*learning_rate_2050*4022700*Capacity_Wind])
# # OPEX
# OPEX = np.array([979800, 340800, 170400])

Year = np.array([2020, 2035, 2050])
CAPEX = np.array([Total_CAPEX_MW*Capacity_Wind, Total_CAPEX_MW*Capacity_Wind*learning_rate_2035, Total_CAPEX_MW*Capacity_Wind*learning_rate_2035*learning_rate_2050])
OPEX = np.array([Avg_OPEX*Capacity_Wind, Avg_OPEX*Capacity_Wind*learning_rate_2035, Avg_OPEX*Capacity_Wind*learning_rate_2035*learning_rate_2050])

a = (r*(1+r)**lftm) / ((1+r)**lftm - 1) # Should this really be constant?

Wind_Costs = np.zeros((5,31))

yearstep = 0
for i in range(31):
    Wind_Costs[0][i] = 2020 + yearstep
    yearstep += 1
    
for i in range(31):
    Wind_Costs[2][i] = lftm
    
# 2020-2035 CAPEX
X = Year[0:2].reshape(-1, 1)
y = CAPEX[0:2]              
model = LinearRegression()
model.fit(X, y)
X_predict = Wind_Costs[0][0:16].reshape(-1, 1) # put the dates of which you want to predict kwh here
Wind_Costs[1][0:16] = model.predict(X_predict)
# 2035-2050 CAPEX
X = Year[1:3].reshape(-1, 1)
y = CAPEX[1:3]
model = LinearRegression()
model.fit(X, y)
X_predict = Wind_Costs[0][16:31].reshape(-1, 1) # put the dates of which you want to predict kwh here
Wind_Costs[1][16:31] = model.predict(X_predict)
# 2020-2035 OPEX
X = Year[0:2].reshape(-1, 1)
y = OPEX[0:2]
model = LinearRegression()
model.fit(X, y)
X_predict = Wind_Costs[0][0:16].reshape(-1, 1) # put the dates of which you want to predict kwh here
Wind_Costs[3][0:16] = model.predict(X_predict)
# 2035-2050 OPEX
X = Year[1:3].reshape(-1, 1)
y = OPEX[1:3]
model = LinearRegression()
model.fit(X, y)
X_predict = Wind_Costs[0][16:31].reshape(-1, 1) # put the dates of which you want to predict kwh here
Wind_Costs[3][16:31] = model.predict(X_predict)

Wind_Costs[4] = a*Wind_Costs[1] + Wind_Costs[3]

def rel_wind_array(SY, NS, TS, WiCo):
    index = np.where(Wind_Costs[0] == SY)[0][0]
    relevant_array = np.zeros(NS)
    for i in range(NS):    
        relevant_array[i] = WiCo[4][index+i*TS]
    
    return relevant_array