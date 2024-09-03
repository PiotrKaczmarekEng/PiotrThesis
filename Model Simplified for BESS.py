# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 15:22:14 2024

@author: spide


Simplified model to fix BESS
"""


# %%  Import library
from gurobipy import *
import os         
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
# import matplotlib.pyplot as plt
import networkx as nx

# %% Model parameters and sets
# Set model name
model = Model('BESS Simple')

# ---- Sets ----



I = [1, 2] # Conv devices (1: Conv, 2: Reconv)
J = [1, 2, 3, 4] # Energy Medium (1: Ammonia, 2: LiquidH2)
K = [1, 2, 3] # Device types (1: Wind, 2: Solar, 3: Elec, 4: Desal)
L = range(2) # Years in time period
N = [1, 2] # Volume based equipment (1: Storage, 2: FPSO)
T = range(len(wind))  # Operational Hours in year


# ---- Parameters ----

BigM = 1000000000000000
E = 0
s0 = 0
smax = 100000000000
smin = 0
alpha = 0.9196588460254348 # [-]
beta = 0.4  # tph
gamma = 50500000 # Wh/t
capacity = 700 # MW
# wind = [ 5, 4, 3, 0, 5, 1]
# solar = [ 1, 2, 1, 1, 0, 0]

# location
latitude = 54.35
longitude = 6.28
area = [longitude, latitude]

wind = func_Wind(area)
solar = func_PV(area)

# Demand 
D = capacity*8760/1000/0.0505   # 121 [kt]
#Converted D into Wh
D = 121000 * gamma /365

PU = 6110500000000/365

# cost
A = [100, 10, 200]

# # Conversion cost
# WC = 122684007.92095348

# # Transport cost
# TC_lj = func_TC(area,E)
# TC = TC_lj[0][E]
    
# %%  ---- Variables ----
# x[k]
x = {}
for k in K:
    x[k] = model.addVar (lb = 0, vtype = GRB.INTEGER, name = 'x[' + str(k) + ']' )

# PU = {}
# for t in T:
#     PU[t] = model.addVar (lb = 0, vtype = GRB.CONTINUOUS, name = 'PU[' + str(t) + ']' )
    
# Power Generated 
PG = {}
for t in T:
    PG[t] = model.addVar (lb = 0, vtype = GRB.CONTINUOUS, name = 'PG[' + str(t) + ']' )

# Battery charging
PC = {}
for t in T:
    PC[t] = model.addVar (lb = 0, vtype = GRB.CONTINUOUS, name = 'PC[' + str(t) + ']' )

# Battery discharging
PD = {}
for t in T:
    PD[t] = model.addVar (lb = 0, vtype = GRB.CONTINUOUS, name = 'PD[' + str(t) + ']' )

# State of charge
s = {}
for t in T:
    s[t] = model.addVar (lb = 0, vtype = GRB.CONTINUOUS, name = 's[' + str(t) + ']' )
    
# Binary charging variable
u = {}
for t in T:
    u[t] = model.addVar (vtype = GRB.BINARY, name = 'u[' + str(t) + ']' )
 
 



# %% Constraints
 
for t in T:
    # model.addConstr(PG[t] == wind[t]*x[1] + solar[t]*x[2])
    model.addConstr(PG[t] == wind[t]*x[1] + solar[t]*x[2])
     

    model.addConstr(PU_array[t] + PC[t] - PD[t] == alpha*PG[t]) # New cons for BESS
    # model.addConstr(PU[t] <= alpha*PG[t]) # New cons for BESS 
    
    model.addConstr(s[t] <= smax) # New cons
    model.addConstr(s[t] >= smin) # new cons
    model.addConstr(PC[t] <= BigM * u[t]) # new cons
    model.addConstr(PD[t] <= BigM * (1-u[t]) ) # new cons
    
    # model.addConstr(PC[t] <= 100000) # new cons
    # model.addConstr(PD[t] <= 100000) # new cons
         
    model.addConstr(PU[t] <= x[3]*beta*gamma) # convert energy to # electrolyzers
    
  
for t in T[1:]:
    model.addConstr(s[t] == s[t-1] + PC[t] - PD[t]) # new cons
model.addConstr(s[0] == s0) # new cons
model.addConstr(PD[0] == s0) # new cons


# model.addConstr(quicksum(PU[t] for t in T) >= D)

model.addConstr(x[3] == 58)# Force # of elec
                                              
# model.addConstr(x[1] == 0)# Force no wind, only solar

#%%

model.update ()
# model.Params.timeLimit = 600
model.update()
model.setParam( 'OutputFlag', True) # silencing gurobi output or not
model.setParam ('MIPGap', 0);       # find the optimal solution
# model.setParam('BranchDir', -1) # Based on tuning
# model.setParam('Presolve', 2) # Based on tuning
# model.write("output2.lp")            # print the model in .lp format file

# ---- Objective Function ----

for l in L:
    model.setObjective( quicksum(A[k-1]*x[k] for k in K))
    model.modelSense = GRB.MINIMIZE
    model.update ()
    model.optimize()


#%%

# Capture the PU values from the initial optimization
# PU_array = np.array([v.x for v in PU.values()])

# plot turbine power output
plt.figure()
plt.plot(np.array(T),wind[0:len(T)])
plt.xlabel('Time')
plt.ylabel('Power in W')
plt.title('Wind: '+str(round(sum(wind)/1000000000,2))+'[GWh]')
plt.show()

# plot solar power output
plt.figure()
plt.plot(np.array(T),solar[0:len(T)])
plt.xlabel('Time')
plt.ylabel('Power in W')
plt.title('Solar '+str(round(sum(solar)/1000000000,2))+'[GWh]')
plt.show()



PG_array = np.array([v.x for v in PG.values()])

# plot power generated wind + solar
plt.figure()
plt.plot(np.array(T), PG_array[0:len(T)])
plt.xlabel('Time')
plt.ylabel('Power in W')
plt.title('PG')
plt.show()

# PU_array = np.array([v.x for v in PU.values()])

# plot power used by electrolyzers
plt.figure()
plt.plot(np.array(T),PU_array[0:len(T)])
plt.xlabel('Time')
plt.ylabel('Power in W')
plt.title('PU')
plt.show()