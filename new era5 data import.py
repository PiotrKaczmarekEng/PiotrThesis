# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 14:19:15 2023

@author: tmell
"""

from feedinlib import era5
import cdsapi
# import matplotlib.pyplot as plt
# import pvlib

#%%

# --- Region and Time Period Prep ---

#singapore 35
#latitude = -52.99
#longitude = 166.92

#singapore 34
#latitude = -49.7
#longitude = 166.74

#US
#latitude = 47.71
#longitude = -47.79

#japan
# latitude = 26.81
# longitude = 126.94

# North-Sea
latitude = 54.35
longitude = 6.28

# Region for study discretized

loclist = []

it = -10

while it < 11:
    loclist.append([latitude+it,longitude+it])
    print([latitude+it,longitude+it])
    # print(i)
    it = it+1
    



# Region for study
LatRegSouth = latitude-10
LatRegNorth = latitude+10
LonRegWest = longitude-10
LonRegEast = longitude+10

latitude = [LatRegSouth, LatRegNorth]  # [latitude south, latitude north]
longitude = [LonRegWest, LonRegEast]  # [longitude west, longitude east]

# # set start and end date (end date will be included
# # in the time period for which data is downloaded)
# start_date, end_date = '2020-01-01', '2020-01-30'
# # set variable set to download (feedinlib: both solar and wind)
# variable = "feedinlib"

# target_file = 'Era 5 test data\ERA5_weather_data_1month_RegNorthSea.nc'

#%%

# get pvlib data for specified region
ds = era5.get_era5_data_from_datespan_and_position(
    variable=variable,
    start_date=start_date, end_date=end_date,
    latitude=latitude, longitude=longitude,
    target_file=target_file)

#%%

# # get windpowerlib data for specified location
# ds = era5.get_era5_data_from_datespan_and_position(
#     variable=variable,
#     start_date=start_date, end_date=end_date,
#     latitude=latitude, longitude=longitude,
#     target_file=target_file)

# # latitude = [0, 90]  # [latitude south, latitude north]
# # longitude = [40, 10]  # [longitude west, longitude east]

# # # get pvlib data for specified area
# # ds = era5.get_era5_data_from_datespan_and_position(
# #     variable=variable,
# #     start_date=start_date, end_date=end_date,
# #     latitude=latitude, longitude=longitude,
# #     target_file=target_file)


# set start and end date (end date will be included
# in the time period for which data is downloaded)
start_date, end_date = '2020-01-01', '2020-09-30'
# set variable set to download
variable = "feedinlib"

target_file = 'Era 5 test data\ERA5_weather_data_NorthSea_010120-300920.nc'

# get feedinlib data (includes pvlib and windpowerlib data)
# for the whole world
ds = era5.get_era5_data_from_datespan_and_position(
    variable="feedinlib",
    start_date=start_date, end_date=end_date,
    target_file=target_file)

# # ds = era5.get_era5_data_from_datespan_and_position(
# #     variable="feedinlib",
# #     start_date='2015-01-01', end_date='2015-12-12',
# #     target_file='ERA5_weather_data.nc')

# # era5_netcdf_filename = 'ERA5_weather_data.nc'

# # area = [13.5, 52.4] #location of production

# # windpowerlib_df = era5.weather_df_from_era5(
# #     era5_netcdf_filename='ERA5_weather_data.nc',
# #     lib='windpowerlib', area=area)

# # pvlib_df = era5.weather_df_from_era5(
# #     era5_netcdf_filename='ERA5_weather_data.nc',
# #     lib='pvlib', area=area)

# # #matplotlib inline
# # pvlib_df.loc[:, ['dhi', 'ghi']].plot(title='Irradiance')
# # plt.xlabel('Time')
# # plt.ylabel('Irradiance in $W/m^2$');


# # feedin = pv_system.feedin(
# #     weather=pvlib_df,
# #     location=(latitude, longitude))



# #https://feedinlib.readthedocs.io/en/features-design-skeleton/load_era5_weather_data.html