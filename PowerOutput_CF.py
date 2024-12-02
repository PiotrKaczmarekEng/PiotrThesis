# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 14:30:00 2024

@author: spide
"""


def func_PV(location):
    
    
    lat_location = location[1]
    lon_location = location[0]
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
    # area = [lon, lat]
    pvlib_df = era5.weather_df_from_era5(     
        era5_netcdf_filename=era5_netcdf_filename,
        lib='pvlib', area=[lon_location, lat_location])
    
    #determining the zenith angle (angle of sun position with vertical in vertical plane) in the specified locations for the time instances downloaded from ERA5
    zenithcalc = pvlib.solarposition.get_solarposition(time=pvlib_df.index,latitude=lat_location, longitude=lon_location, altitude=None, pressure=None, method='nrel_numpy', temperature=pvlib_df['temp_air'])
    
    #determining DNI from GHI, DHI and zenith angle for the time instances downloaded from ERA5
    dni = pvlib.irradiance.dni(pvlib_df['ghi'],pvlib_df['dhi'], zenith=zenithcalc['zenith'], clearsky_dni=None, clearsky_tolerance=1.1, zenith_threshold_for_zero_dni=88.0, zenith_threshold_for_clearsky_limit=80.0)
    
    #adding DNI to dataframe with PV weather data
    pvlib_df['dni'] = dni
    
    #replacing 'NAN' in DNI column with 0 to prevent 'holes' in graphs (NANs are caused by the zenith angle being larger than the 'zenith threshold for zero dni' angle set with GHI and DHI not yet being zero. The DNI should be 0 in that case)
    pvlib_df['dni'] = pvlib_df['dni'].fillna(0)
    
    #determining PV power generation
    # location = (lat,lon)
    feedin = pv_system.feedin(
        weather=pvlib_df,
        location=(lat_location, lon_location))
    
    feedinarray = multsolar*feedin.values #hourly energy production over the year of 1 solar platform of the specified kind in the specified location
    feedinarray[feedinarray<0]=0
    
    #plotting PV power generation
    plt.figure()
    feedin.plot(title='PV feed-in '+str(lat_location)+', '+str(lon_location))
    plt.xlabel('Time')
    plt.ylabel('Power in W');
    
    feedin_index = feedin.index
    
    return feedinarray    


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
    
    # title = 'Wind turbine power production at ' + str(location)
    # # plot turbine power output
    # plt.figure()
    # my_turbine.power_output.plot(title=title)
    # plt.xlabel('Time')
    # plt.ylabel('Power in W')
    # plt.show()
    
    my_turbinearray = multwind*my_turbine.power_output.values #hourly energy production over the year of 1 wind turbine of the specified kind in the specified location
    
    return my_turbinearray

PowerList = []
for vert in range(size):
    for loc in loc_matrix[vert]:
        print("Location: ",loc)
        print("Solar Power: ",sum(func_PV(loc)))
        print("Wind Power: ",sum(func_Wind(loc)))
        PowerList.append(sum(func_PV(loc))+sum(func_Wind(loc)))
        
df_relevant_lat = pd.DataFrame(columns=['longitude','latitude','Power Output','Capacity Factor'])
counter=0
for vert in range(size):
    counter2=0
    for j in range(size):
        
        df_relevant_lat.loc[counter,'longitude'] = loc_matrix[vert][counter2][0]
        df_relevant_lat.loc[counter,'latitude'] = loc_matrix[vert][counter2][1]
        df_relevant_lat.loc[counter,'Power Output'] = PowerList[counter]
        df_relevant_lat.loc[counter,'Capacity Factor'] = PowerList[counter]/82357824000
        counter2 = counter2 + 1     
        counter = counter + 1      

#%%
fig = px.density_mapbox(df_relevant_lat, lat = 'latitude', lon = 'longitude', z = 'Capacity Factor',
                        radius = 15,
                        center = dict(lat = latitude, lon = longitude),
                        zoom = 3,
                        mapbox_style = 'open-street-map',
                        title = title_str,
                        color_continuous_scale = 'magma')

# Adjust color of heatmap by adding more points for density
fig.add_trace(
    go.Scattermapbox(
        lat=df_relevant_lat["latitude"],
        lon=df_relevant_lat["longitude"],
        mode="markers",
        showlegend=False,
        hoverinfo="skip",
        marker={
            "color": df_relevant_lat["Capacity Factor"],
            "size": df_relevant_lat['Capacity Factor'].fillna(0).infer_objects(copy=False),
            "coloraxis": "coloraxis",
            # desired max size is 15. see https://plotly.com/python/bubble-maps/#united-states-bubble-map
            "sizeref": (df_relevant_lat['Capacity Factor'].max()) / 15 ** 2,
            "sizemode": "area",
        },
    )
)

pio.renderers.default='browser'
fig.show()

#%%

import matplotlib.pyplot as plt

data=df_relevant_lat['Capacity Factor']

latcorners = df_relevant_lat['latitude'][:]
loncorners = df_relevant_lat['longitude'][:]
lon_0 = df_relevant_lat['longitude']
lat_0 = df_relevant_lat['latitude']

# create figure and axes instances
fig = plt.figure(figsize=(8,8))
ax = fig.add_axes([0.1,0.1,0.8,0.8])

lon_0
lat_0

m = Basemap(projection='stere',lon_0=,lat_0=,lat_ts=lat_0,\
            llcrnrlat=latcorners[0],urcrnrlat=latcorners[35],\
            llcrnrlon=loncorners[0],urcrnrlon=loncorners[5],\
            rsphere=6371200.,resolution='l',area_thresh=10000)

m.drawcoastlines()    
m.drawcountries()

ny = data.shape[0]; nx = data.shape[1]
lons, lats = m.makegrid(nx, ny) # get lat/lons of ny by nx evenly space grid.
x, y = m(lons, lats) # compute map proj coordinates.

# draw filled contours.
clevs = [0,1,2.5,5,7.5,10,15,20,30,40,50,70,100,150,200,250,300,400,500,600,750]
cs = m.contourf(x,y,data,clevs,cmap=cm.s3pcpn)

plt.show()