# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:05:31 2024

@author: spide
"""
#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopy.distance
import math
from openpyxl import load_workbook
from feedinlib import era5
import pvlib
from feedinlib import Photovoltaic
from feedinlib import get_power_plant_data
from windpowerlib.modelchain import ModelChain
from windpowerlib.wind_turbine import WindTurbine
from windpowerlib import get_turbine_types
from datetime import timedelta
from gurobipy import *

#%%

latitude =  26.81
longitude = 126.94

multsolar = 524*1000/206.7989   # Installed capacity per unit [kWp] * 1000 (to make it MW) / capacity from PVlib [kWp]


# set start and end date (end date will be included
# in the time period for which data is downloaded)
start_date, end_date = '2020-01-01', '2020-01-30'
# set variable set to download
variable = 'feedinlib'

era5_netcdf_filename = 'Era 5 test data\ERA5_weather_data_test_RegTokyo.nc' #referring to file with weather data downloaded earlier using the ERA5 API


#%%

# Define the dimensions of the matrix
size = 21  # This will create a 21x21 matrix
# Calculate the range for rows and columns
start = -10
end = 10
# Create a matrix of the specified size, with each element being a tuple of length 2
loc_matrix = np.empty((size, size), dtype=object)
for i in range(size):
    for j in range(size):
        loc_matrix[i,j] = (start + j + longitude, end - i + latitude)


area = [longitude,latitude]

#%%
# --- Solar feedin function ---
# Function which returns the feedin power in [MW], based on input location [longitude,latitude]
def func_PV(location):
    
    # get modules
    module_df = get_power_plant_data(dataset='SandiaMod') #retrieving dataset for PV modules
    
    # get inverter data
    inverter_df = get_power_plant_data(dataset='cecinverter') #retrieving dataset for inverters
    
    temp_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_polymer'] #defining temperature model parameters (leaving at default causes errors)
    
    #PV system definition
    system_data = {
        'module_name': 'Advent_Solar_Ventura_210___2008_',  # module name as in database
        'inverter_name': 'ABB__MICRO_0_25_I_OUTD_US_208__208V_',  # inverter name as in database
        'azimuth': 180, #angle of sun position with north in horizontal plane
        'tilt': 30, #angle of solar panels with horizontal
        'albedo': 0.075, #albedo (fraction of sun light reflected) of ocean water (https://geoengineering.global/ocean-albedo-modification/)
        'temperature_model_parameters': temp_params,
    }
    
    pv_system = Photovoltaic(**system_data)
    
    #getting the needed weather data for PV calculations from the file as downloaded with a seperate script from ERA5
    pvlib_df = era5.weather_df_from_era5(     
        era5_netcdf_filename=era5_netcdf_filename,
        lib='pvlib', area=location)
    
    #determining the zenith angle (angle of sun position with vertical in vertical plane) in the specified locations for the time instances downloaded from ERA5
    zenithcalc = pvlib.solarposition.get_solarposition(time=pvlib_df.index,latitude=latitude, longitude=longitude, altitude=None, pressure=None, method='nrel_numpy', temperature=pvlib_df['temp_air'])
    
    #determining DNI from GHI, DHI and zenith angle for the time instances downloaded from ERA5
    dni = pvlib.irradiance.dni(pvlib_df['ghi'],pvlib_df['dhi'], zenith=zenithcalc['zenith'], clearsky_dni=None, clearsky_tolerance=1.1, zenith_threshold_for_zero_dni=88.0, zenith_threshold_for_clearsky_limit=80.0)
    
    #adding DNI to dataframe with PV weather data
    pvlib_df['dni'] = dni
    
    #replacing 'NAN' in DNI column with 0 to prevent 'holes' in graphs (NANs are caused by the zenith angle being larger than the 'zenith threshold for zero dni' angle set with GHI and DHI not yet being zero. The DNI should be 0 in that case)
    pvlib_df['dni'] = pvlib_df['dni'].fillna(0)
    
    #determining PV power generation
    feedin = pv_system.feedin(
        weather=pvlib_df,
        location=(location[1], location[0]))
    
    feedinarray = multsolar*feedin.values #hourly energy production over the year of 1 solar platform of the specified kind in the specified location
    feedinarray[feedinarray<0]=0
    
    return feedinarray


#%%

# --- Wind function ---

def func_Wind(location):
    
    df = get_turbine_types(print_out=False)
    
    #defining the wind turbine system
    turbine_data= {
        "turbine_type": "E-101/3050",  # turbine type as in register
        "hub_height": 130,  # in m
    }
    my_turbine = WindTurbine(**turbine_data)
    
    #getting the needed weather data for PV calculations from the file as downloaded with a seperate script from ERA5
    windpowerlib_df = era5.weather_df_from_era5(
        era5_netcdf_filename=era5_netcdf_filename,
        lib='windpowerlib', area=area)
    
    #increasing the time indices by half an hour because then they match the pvlib time indices so the produced energy can be added later
    #assumed wind speeds half an hour later are similar and this will not affect the results significantly
    windpowerlib_df.index = windpowerlib_df.index + timedelta(minutes=30)
    
    # power output calculation for e126
    
    # own specifications for ModelChain setup
    modelchain_data = {
        'wind_speed_model': 'logarithmic',      # 'logarithmic' (default),
                                                # 'hellman' or
                                                # 'interpolation_extrapolation'
        'density_model': 'ideal_gas',           # 'barometric' (default), 'ideal_gas'
                                                #  or 'interpolation_extrapolation'
        'temperature_model': 'linear_gradient', # 'linear_gradient' (def.) or
                                                # 'interpolation_extrapolation'
        'power_output_model': 'power_curve',    # 'power_curve' (default) or
                                                # 'power_coefficient_curve'
        'density_correction': True,             # False (default) or True
        'obstacle_height': 0,                   # default: 0
        'hellman_exp': None}                    # None (default) or None
    
    # initialize ModelChain with own specifications and use run_model method to
    # calculate power output
    mc_my_turbine = ModelChain(my_turbine, **modelchain_data).run_model(windpowerlib_df)
    # write power output time series to WindTurbine object
    my_turbine.power_output = mc_my_turbine.power_output
    
    
    # # plot turbine power output
    # plt.figure()
    # my_turbine.power_output.plot(title='Wind turbine power production')
    # plt.xlabel('Time')
    # plt.ylabel('Power in W')
    # plt.show()
    
    my_turbinearray = multwind*my_turbine.power_output.values #hourly energy production over the year of 1 wind turbine of the specified kind in the specified location
    
    return my_turbinearray



#%%

# --- Transport Cost Function ---    


# The output of this function gives the transport cost for each year, and each medium
def func_TC(location):
    
    location = (location[1],location[0])
    distancesea = distanceseafactor*geopy.distance.geodesic(location, coords_port).km
    
    TC = []
    for j in J:
        TCyear = Xtransport*Cbasetransport[j-1] + Xkmsea*Ckmsea[j-1] + Xkmland*Ckmland
        TC.append(TCyear)
    
    return TC
    




#%%
feedinarray = func_PV(area)

size_feedin = len(feedinarray)



func_PV(list(loc_matrix[10][10]))


# PV Feedins for all at the same latitude, with different longitudes
list_of_feedins = []
for i in range(21):
    print(loc_matrix[0][i])
    temp_feedin = func_PV(list(loc_matrix[0][i]))
    print(temp_feedin[0])
    list_of_feedins.append(temp_feedin)




feedin_matrix = np.empty((size, size, size_feedin), dtype=object)
# i: long, j: lat, k: hour
for i in range(size):
    for j in range(size):
        for k in range(size_feedin):
            feedin_matrix[j,i,k] = (loc_matrix[i][j][0],loc_matrix[i][j][1],feedinarray[k])
            
