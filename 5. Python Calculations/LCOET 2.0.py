# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 11:00:48 2024

@author: spide
"""

import numpy as np
import matplotlib.pyplot as plt

# Transport Cost Pipeline computation

#### Input params

dist_set = np.linspace(0, 1480, num=38)
Capacity = 1 # [GW/hr]
HoursYear = 365*24 # [hr]
EnergyProduced = 5160 # [GWh]
CapacityFactor = EnergyProduced/HoursYear # ~60% CF

EnergyTransported = 5160 # [GWh/yr]
CAPEX_pipe = 0.12 # MUSD/GW/km
OPEX_pipe = 0.8 # % of CAPEX each year
r = 0.02 # Interest Rate [%]
pipe_years = 25

### Relevant Params for Ammonia pipeline
UtilisationFactor = 0.75
DiameterPipe = 10 #[cm]
PipeCapacity = 15.245*((DiameterPipe/5) - 1)**1.6468 # ton NH3 per hour
PipeCapacityYearly = PipeCapacity*HoursYear*UtilisationFactor #tonNH3/year
PipeCapacityYearly

Capacity = 0.5# [GW/yr]
HoursYear = 365*24 # [hr]
EnergyProduced =  Capacity*HoursYear# [GWh]
# CapacityFactor = EnergyProduced/HoursYear # ~60% CF

EnergyTransported = EnergyProduced # [GWh/yr]
CAPEX_pipe = # MillionEuro/GW/km
OPEX_pipe = 2# [%] of CAPEX each year
r = 0.08 # Interest Rate [%]
pipe_years = 30


def LCOET_func(dist):
    CostPipeYear = np.zeros((4,25))
    
    # Year row
    for n in range(pipe_years):
        DF = 1/((1+r)**(n+1))
        CostPipeYear[0][n] = n+1
    # CAPEX row
        CostPipeYear[1][n] = CAPEX_pipe*dist*1000000 # [USD]
    # OPEX row
        CostPipeYear[2][n] = (OPEX_pipe*0.01*CostPipeYear[1][n]) * DF #[USD/yr]
    # Denominator row
        CostPipeYear[3][n] = EnergyTransported / DF # [GWh/yr]
    
    LCOET = (CostPipeYear[1][0] + sum(CostPipeYear[2])) / sum(CostPipeYear[3]) #[USD/GWh]
    LCOET = LCOET/1000 # [USD/MWh]
    print('CAPEX = ', CostPipeYear[1][0])
    print('Sum OPEX = ', sum(CostPipeYear[2]))
    print('Sum Energy = ', sum(CostPipeYear[3]))
    print('LCOET = ', LCOET)
    return LCOET

LCOET_array = np.zeros(38)
step = 0
for dist in dist_set:
    LCOET_array[step] = LCOET_func(dist)
    step += 1

plt.plot(dist_set, LCOET_array, marker='o', linestyle='-', color='b')
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
plt.xlim(0,1500)
plt.title('LCOET vs Distance')
plt.xlabel('Distance [km]')
plt.ylabel('LCOET [USD/MWh]')
plt.grid(True)
plt.show()
