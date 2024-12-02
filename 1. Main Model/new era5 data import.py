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

#%%

# set start and end date (end date will be included
# in the time period for which data is downloaded)
start_date, end_date = '2021-01-01', '2021-09-30'
# set variable set to download
variable = "feedinlib"

target_file = 'Era 5 test data\ERA5_weather_data_NorthSea_010121-300921.nc'

# get pvlib data for specified region
ds = era5.get_era5_data_from_datespan_and_position(
    variable=variable,
    start_date=start_date, end_date=end_date,
    latitude=latitude, longitude=longitude,
    target_file=target_file)

#%%

# set start and end date (end date will be included
# in the time period for which data is downloaded)
start_date, end_date = '2022-01-01', '2022-09-30'
# set variable set to download
variable = "feedinlib"

target_file = 'Era 5 test data\ERA5_weather_data_NorthSea_010122-300922.nc'


# get pvlib data for specified region
ds = era5.get_era5_data_from_datespan_and_position(
    variable=variable,
    start_date=start_date, end_date=end_date,
    latitude=latitude, longitude=longitude,
    target_file=target_file)

# # get feedinlib data (includes pvlib and windpowerlib data)
# # for the whole world
# ds = era5.get_era5_data_from_datespan_and_position(
#     variable="feedinlib",
#     start_date=start_date, end_date=end_date,
#     target_file=target_file)



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

#%%


### CDS-Beta

import cdsapi


target_file = 'Era 5 test data\ERA5_weather_data_NorthSea_010122-300922.nc'
variable = [
    "100u",
    "100v",
    "fsr",
    "sp",
    "fdir",
    "ssrd",
    "2t",
    "10u",
    "10v",
]
request = {
    "format": "netcdf",
    "product_type": "reanalysis",
    "time": [
        "00:00",
        "01:00",
        "02:00",
        "03:00",
        "04:00",
        "05:00",
        "06:00",
        "07:00",
        "08:00",
        "09:00",
        "10:00",
        "11:00",
        "12:00",
        "13:00",
        "14:00",
        "15:00",
        "16:00",
        "17:00",
        "18:00",
        "19:00",
        "20:00",
        "21:00",
        "22:00",
        "23:00",
    ],
}


client = cdsapi.Client()

dataset = 'reanalysis-era5-single-levels'
request = {
    'product_type': ['reanalysis'],
    'variable': variable,
    'year': ['2024'],
    'month': ['03'],
    'day': ['01'],
    'time': ['13:00'],
    'pressure_level': ['1000'],
    'data_format': 'grib',
}
target = target_file

client.retrieve(dataset, request, target)

#%%
import cdsapi
 
c = cdsapi.Client()
 
first_year = 2018
last_year = 2019


for year in range(first_year, last_year + 1):
    for month in range(1, 13):
        print("=========================================================")
        print("Downloading {year}-{month:02d}".format(year=year, month=month))
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': '2m_temperature',
                'year': str(year),
                'month': "{month:02d}".format(month=month),
                'day': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                    '13', '14', '15',
                    '16', '17', '18',
                    '19', '20', '21',
                    '22', '23', '24',
                    '25', '26', '27',
                    '28', '29', '30',
                    '31',
                ],
                'time': [
                    '00:00', '01:00', '02:00',
                    '03:00', '04:00', '05:00',
                    '06:00', '07:00', '08:00',
                    '09:00', '10:00', '11:00',
                    '12:00', '13:00', '14:00',
                    '15:00', '16:00', '17:00',
                    '18:00', '19:00', '20:00',
                    '21:00', '22:00', '23:00',
                ],
                'area': [
                    90, 170, 80,
                    180,
                ],
                'format': 'grib',
            },
            "{year}-{month:02d}.grib".format(year=year, month=month))