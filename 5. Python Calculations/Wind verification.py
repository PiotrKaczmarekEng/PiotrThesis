# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 23:58:54 2024

@author: spide
"""
# 24 elec_ low wind
lon = 7.08
lat = 53.55

# 23 elec, medium wind
lon = 7.08
lat = 53.95

# 22 elec, high wind
lon = 7.08
lat = 54.15

area = [lon,lat]

def func_Wind(location):
    
    lat_location = location[1]
    lon_location = location[0]
    
    df = get_turbine_types(print_out=False)
    
    #defining the wind turbine system
    turbine_data= {
        "turbine_type": "E-101/3050",  # turbine type as in register
        "hub_height": 130,  # in m
    }
    my_turbine = WindTurbine(**turbine_data)
    
    #getting the needed weather data for wind power calculations from the file as downloaded with a seperate script from ERA5
    # [lon,lat]
    windpowerlib_df = era5.weather_df_from_era5(
        era5_netcdf_filename=era5_netcdf_filename,
        lib='windpowerlib', area=[lon_location, lat_location])
    
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
    
    title = 'Wind turbine power production at (' + str([lat_location, lon_location])+')'
    # plot turbine power output
    plt.figure()
    my_turbine.power_output.plot(title=title)
    plt.xlabel('Time')
    plt.ylabel('Wind Power in [W]')
    plt.show()
    
    my_turbinearray = multwind*my_turbine.power_output.values #hourly energy production over the year of 1 wind turbine of the specified kind in the specified location
    
    return my_turbinearray


# 24 elec_ low wind
lon = 7.08
lat = 53.55

# 23 elec, medium wind
lon = 7.08
lat = 53.95

# 22 elec, high wind
lon = 7.08
lat = 54.15

area_low = [7.08, 53.55]
area_med = [7.08, 53.95]
area_high = [7.08, 54.15]

low_wind = func_Wind(area_low).sum()
med_wind = func_Wind(area_med).sum()
high_wind = func_Wind(area_high).sum()

pd.dataframe