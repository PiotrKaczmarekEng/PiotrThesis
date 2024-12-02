# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 11:17:09 2024

@author: spide
"""

from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Wind turbine costs

# Startyear = 2047
learning_rate_2035 = 40/115
learning_rate_2050 = 0.5
#%%
# CAPEX [Eur/kW] in 2020
Development = 212
Turbine = 1060
Plant_Balance = 350
# Installation = included in turbine cost
Decommission = 44
Total_CAPEX = Development + Turbine + Plant_Balance + Decommission

Total_CAPEX_MW = Total_CAPEX * 1000

#%%
# OPEX [Eur/MW] in 2020
Avg_OPEX = 135000

#%% Combined table (training data)

Capacity_Wind = 12 # [MW]

# #  TYCHOS NUMBERS (FLOATING)
# # Year
# Year = np.array([2025, 2035, 2050])
# # CAPEX
# CAPEX = np.array([4022700*Capacity_Wind, learning_rate_2035*4022700*Capacity_Wind, learning_rate_2035*learning_rate_2050*4022700*Capacity_Wind])
# # OPEX
# OPEX = np.array([979800, 340800, 170400])

# MY NUMBERS (FIXED) (still with old assumption factors for learning rate)

# Year
Year = np.array([2020, 2035, 2050])
# CAPEX
CAPEX = np.array([Total_CAPEX_MW*Capacity_Wind, Total_CAPEX_MW*Capacity_Wind*learning_rate_2035, Total_CAPEX_MW*Capacity_Wind*learning_rate_2035*learning_rate_2050])
# OPEX
OPEX = np.array([Avg_OPEX*Capacity_Wind, Avg_OPEX*Capacity_Wind*learning_rate_2035, Avg_OPEX*Capacity_Wind*learning_rate_2035*learning_rate_2050])

#%%
r = 0.08 # Interest rate
l = 25
a = (r*(1+r)**l) / ((1+r)**l - 1) # Should this really be constant?

#%%
Wind_Costs = np.zeros((5,31))

yearstep = 0
for i in range(31):
    Wind_Costs[0][i] = 2020 + yearstep
    yearstep += 1
    
for i in range(31):
    Wind_Costs[2][i] = l
  

#  5 rows, 31 cols
# row 0: headers of year (2020-2050)
# row 1: CAPEX
# row 2: Lifetime
# row 3: OPEX
# row 4: total yearly


#%% Regression CAPEX

# 2020-2035
X = Year[0:2].reshape(-1, 1)
y = CAPEX[0:2]
              
# X = np.array([2025, 2035, 2050]).reshape(-1, 1)  # put your dates in here
# y = np.array([4022700*12, (40/115)*4022700*12, 0.5*(40/115)*4022700*12])  # put your cost in here

model = LinearRegression()
model.fit(X, y)

X_predict = Wind_Costs[0][0:16].reshape(-1, 1) # put the dates of which you want to predict kwh here
Wind_Costs[1][0:16] = model.predict(X_predict)

# 2035-2050
X = Year[1:3].reshape(-1, 1)
y = CAPEX[1:3]

model = LinearRegression()
model.fit(X, y)

X_predict = Wind_Costs[0][16:31].reshape(-1, 1) # put the dates of which you want to predict kwh here
Wind_Costs[1][16:31] = model.predict(X_predict)

# Generate predictions for the years from 2020 to 2050
years = np.arange(2020, 2051).reshape(-1, 1)
predictions = Wind_Costs[1]

# Print the predictions
for year, pred in zip(years.flatten(), predictions):
    print(f"Year: {year}, Predicted cost: {pred}")

# Plotting the data and the regression line
plt.scatter(Year, CAPEX, color='blue', label='Actual data')
plt.plot(years, predictions, color='red', label='Regression line')
plt.xlabel('Year')
plt.ylabel('Cost')
plt.title('Linear Regression - Wind CAPEX')
plt.legend()
plt.show()


#%% Regression OPEX

# 2020-2035
X = Year[0:2].reshape(-1, 1)
y = OPEX[0:2]
              
# X = np.array([2025, 2035, 2050]).reshape(-1, 1)  # put your dates in here
# y = np.array([4022700*12, (40/115)*4022700*12, 0.5*(40/115)*4022700*12])  # put your cost in here

model = LinearRegression()
model.fit(X, y)

X_predict = Wind_Costs[0][0:16].reshape(-1, 1) # put the dates of which you want to predict kwh here
Wind_Costs[3][0:16] = model.predict(X_predict)

# 2035-2050
X = Year[1:3].reshape(-1, 1)
y = OPEX[1:3]

model = LinearRegression()
model.fit(X, y)

X_predict = Wind_Costs[0][16:31].reshape(-1, 1) # put the dates of which you want to predict kwh here
Wind_Costs[3][16:31] = model.predict(X_predict)

# Generate predictions for the years from 2020 to 2050
years = np.arange(2020, 2051).reshape(-1, 1)
predictions = Wind_Costs[3]

# Print the predictions
for year, pred in zip(years.flatten(), predictions):
    print(f"Year: {year}, Predicted cost: {pred}")

# Plotting the data and the regression line
plt.scatter(Year, OPEX, color='blue', label='Actual data')
plt.plot(years, predictions, color='red', label='Regression line')
plt.xlabel('Year')
plt.ylabel('Cost')
plt.title('Linear Regression - Wind OPEX')
plt.legend()
plt.show()

#%% Total yearly CAPEX (full table 2020-2050)

Wind_Costs[4] = a*Wind_Costs[1] + Wind_Costs[3]


#%% Only relevant years

Startyear = 2047
Nsteps = 2
timestep = 3

# timeperiod = 30 #time period of simulation in years
# timestep = 3 #time step of simulation in years
# Nsteps = int(timeperiod/timestep+1) #number of time steps (important for data selection from excel and loop at the end)
def rel_wind_array(SY, NS, TS, WiCo):
    index = np.where(Wind_Costs[0] == SY)[0][0]
    relevant_array = np.zeros(NS)
    for i in range(NS):    
        relevant_array[i] = WiCo[4][index+i*TS]
    
    return relevant_array

rel_wind_array(Startyear, Nsteps, timestep, Wind_Costs)

# index = np.where(Wind_Costs[0] == Startyear)[0][0]
# years_relevant = []
# Cost_relevant = []
# for i in range(Nsteps):
#     print(i)
#     print(Wind_Costs[0][index+i*timestep])
#     years_relevant.append(Wind_Costs[0][index+i*timestep])
#     print(Wind_Costs[4][index+i*timestep])
#     Cost_relevant.append(Wind_Costs[4][index+i*timestep])
#     indices.append(Wind_Costs[0][index+i*timestep])


# timeperiod
# timestep
# Nsteps



#%% CLEANED UP

#### Input params

Startyear = 2047
Nsteps = 2
timestep = 3
learning_rate_2035 = 40/115
learning_rate_2050 = 0.5
Capacity_Wind = 12 
# CAPEX [Eur/kW] in 2020
Development = 212
Turbine = 1060
Plant_Balance = 350
Decommission = 44
Total_CAPEX = Development + Turbine + Plant_Balance + Decommission
Total_CAPEX_MW = Total_CAPEX * 1000
# OPEX [Eur/MW] in 2020
Avg_OPEX = 135000
r = 0.08 # Interest rate
lftm = 25


#### Computation

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

def rel_wind_array(SY, NS, TS, WiCo):
    index = np.where(Wind_Costs[0] == SY)[0][0]
    relevant_array = np.zeros(NS)
    for i in range(NS):    
        relevant_array[i] = WiCo[4][index+i*TS]
    
    return relevant_array

rel_wind_array(Startyear, Nsteps, timestep, Wind_Costs)