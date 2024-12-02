# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 12:52:18 2024

@author: spide
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 11:17:09 2024

@author: spide
"""

from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Ammonia Conversion and 
# Conversion costs

# Tycho's numbers
r = 0.08 # Interest rate [%/100]
lftm = 25 #lifetime [years]
a = (r*(1+r)**lftm) / ((1+r)**lftm - 1)

ThroughputConv = 100000/365 #tonNH3/day
ThroughputReconv = 1200 #tonNH3/day
TotalDirectCostConv = 80 #million euro
TotalDirectCostReconv = 63 #million euro
FactorConv = (TotalDirectCostConv/TotalDirectCostReconv)*(ThroughputReconv/ThroughputConv)

# Reconv
CAPEX_1 = 94500000 # euro
OPEX_1 = 1600000 # euro/yr
Capacity_reconv = 78840 # [tonH2/yr]
Capacity_used_device = 10000

Rec_Costs = np.zeros((5,31))

yearstep = 0

for i in range(31):
    Rec_Costs[0][i] = 2020 + yearstep # Year
    Rec_Costs[1][i] = CAPEX_1*Capacity_used_device/Capacity_reconv # CAPEX
    Rec_Costs[2][i] = lftm # Lifetime
    Rec_Costs[3][i] = OPEX_1*Capacity_used_device/Capacity_reconv # OPEX
    Rec_Costs[4][i] = Rec_Costs[1][i]*a+Rec_Costs[3][i] # Total = CAPEX*a + OPEX
    yearstep += 1
    
# Conv

Con_Costs = np.zeros((5,31))

yearstep = 0

for i in range(31):
    Con_Costs[0][i] = 2020 + yearstep # Year
    Con_Costs[1][i] = FactorConv*Rec_Costs[1][i] # CAPEX
    Con_Costs[2][i] = lftm # Lifetime
    Con_Costs[3][i] = FactorConv*Rec_Costs[3][i] # OPEX
    Con_Costs[4][i] = Con_Costs[1][i]*a+Con_Costs[3][i] # Total = CAPEX*a + OPEX
    yearstep += 1

#%% LH2 Conversion and Reconversion

from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Conversion
# Tycho's numbers
r = 0.08 # Interest rate [%/100]
lftm = 30   # Conversion device lifetime
a = (r*(1+r)**lftm) / ((1+r)**lftm - 1)
Prod_conv_unit = 10000 # [tonsH2/yr] yearly production of 1 conversion unit

# Eur/ton/yr
LH2_data = np.array([
    [2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050],
    [5623.93, 5567.69, 5511.45, 5455.21, 5398.97, 5342.74, 5286.50, 5230.26, 5174.02, 5117.78, 5061.54, 5010.92, 4960.31, 4909.69, 4859.08, 4808.46, 4757.85, 4707.23, 4656.62, 4606.00, 4555.38, 4504.77, 4454.15, 4403.54, 4352.92, 4302.31, 4251.69, 4201.08, 4150.46, 4099.85, 4049.23],
    [224.96, 222.71, 220.46, 218.21, 215.96, 213.71, 211.46, 209.21, 206.96, 204.71, 202.46, 200.44, 198.41, 196.39, 194.36, 192.34, 190.31, 188.29, 186.26, 184.24, 182.22, 180.19, 178.17, 176.14, 174.12, 172.09, 170.07, 168.04, 166.02, 163.99, 161.97]
])

LH2_Con_Costs = np.zeros((5,31))
LH2_Con_Costs[0] = LH2_data[0]                   #Year
LH2_Con_Costs[1] = LH2_data[1]*Prod_conv_unit    #CAPEX [Euro/yr]
LH2_Con_Costs[2] = lftm                          #Lifetime [Years]
LH2_Con_Costs[3] = LH2_data[2]*Prod_conv_unit    #OPEX [Euro/yr]
LH2_Con_Costs[4] = LH2_Con_Costs[1]*a+LH2_Con_Costs[3]             #Total yearly [Euro/yr]


# Reconversion
FactorConvLH2 = 0.2

LH2_Rec_Costs = np.zeros((5,31))

yearstep = 0
for i in range(31):
    LH2_Rec_Costs[0][i] = LH2_Con_Costs[0][i]                   #Year
    LH2_Rec_Costs[1][i] = FactorConvLH2*LH2_Con_Costs[1][i]     #CAPEX [Euro/yr]
    LH2_Rec_Costs[2][i] = lftm                                  #Lifetime [years]
    LH2_Rec_Costs[3][i] = FactorConvLH2*LH2_Con_Costs[3][i]     #OPEX [Euro/yr]
    LH2_Rec_Costs[4][i] = FactorConvLH2*LH2_Con_Costs[4][i]     #Total yearly [Euro/yr]
    yearstep += 1
    
# LH2_Con_Costs == LH2_Rec_Costs

def rel_con_arrayLH2(SY, NS, TS, ConLH2Co):
    index = np.where(LH2_Con_Costs[0] == SY)[0][0]
    relevant_array_con_LH2 = np.zeros(NS)
    for i in range(NS):    
        relevant_array_con_LH2[i] = ConLH2Co[4][index+i*TS]
    
    return relevant_array_con_LH2

def rel_rec_arrayLH2(SY, NS, TS, RecLH2Co):
    index = np.where(LH2_Rec_Costs[0] == SY)[0][0]
    relevant_array_rec_LH2 = np.zeros(NS)
    for i in range(NS):    
        relevant_array_rec_LH2[i] = RecLH2Co[4][index+i*TS]
    
    return relevant_array_rec_LH2

rel_con_arrayLH2(Startyear, Nsteps, timestep, LH2_Con_Costs)
rel_rec_arrayLH2(Startyear, Nsteps, timestep, LH2_Rec_Costs)
