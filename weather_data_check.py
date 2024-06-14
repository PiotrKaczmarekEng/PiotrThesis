# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 15:02:40 2024

@author: spide
"""

# NetCDF view to check weather data

import netCDF4 as nc
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# Load the NetCDF file
file_path = 'Era 5 test data\ERA5_weather_data_test_RegTokyo_Corrected.nc'


# Using netCDF4 to read the file
dataset = nc.Dataset(file_path)

# Print the dataset information
print(dataset)

# Using xarray for easier manipulation
data = xr.open_dataset(file_path)

# Print the data variables and coordinates
print(data)

# Select a specific variable to plot (assuming a variable named 'temperature' exists)
variable = data['u10']

# Plotting the data (assuming the variable is 2D for simplicity)
variable.plot()

# Show the plot
plt.show()


#%%

latitude =  26.81
longitude = 126.94

era5_netcdf_filename = 'Era 5 test data\ERA5_weather_data_test_RegTokyo.nc'


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
        
#%%



# Access the variables
u10 = data['u10']
v10 = data['v10']

# Print variable information
print(u10)
print(v10)

# Example: Plotting u10 and v10 at a specific time step
time_step = 0  # adjust based on the time dimension

# Select data at a specific time step
u10_time_slice = u10.isel(time=time_step)
v10_time_slice = v10.isel(time=time_step)

# Plot u10 and v10 at the specific time step
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

u10_time_slice.plot(ax=axes[0])
axes[0].set_title('u10 at time step {}'.format(time_step))

v10_time_slice.plot(ax=axes[1])
axes[1].set_title('v10 at time step {}'.format(time_step))

plt.show()

# # Example: Plotting how u10 changes over time at a specific location
# latitude_index = loc_matrix[0][0][1]  # adjust based on the latitude dimension
# longitude_index = loc_matrix[0][0][0]  # adjust based on the longitude dimension

list_loc = [loc_matrix[0][0], loc_matrix[0][20], loc_matrix[20][0], loc_matrix[20][20]]

#%%

# Example: Plotting how u10 changes over time at a specific location
for i in range(4):
    latitude_value = list_loc[i][1]  # example latitude value
    longitude_value = list_loc[i][0]  # example longitude value
    print('iter: ', i+1, '--- latitude_value: ', latitude_value,', longitude_value: ', longitude_value)
    # Find the nearest indices for the given latitude and longitude
    lon_idx = u10.longitude.sel(longitude=longitude_value, method="nearest").values
    lat_idx = u10.latitude.sel(latitude=latitude_value, method="nearest").values
    # lon_idx = u10.longitude.sel(longitude=longitude_value, method="nearest").values
    print('iter: ', i+1, '--- lat_idx: ', lat_idx,', lon_idx: ', lon_idx)

    
    
    # Select data at the nearest latitude and longitude
    u10_location = u10.sel(latitude=lat_idx, longitude=lon_idx)
    v10_location = v10.sel(latitude=lat_idx, longitude=lon_idx)
    
    # Plot u10 and v10 over time at the specific location
    fig, ax = plt.subplots(figsize=(10, 5))
    
    u10_location.plot(ax=ax, label='u10')
    v10_location.plot(ax=ax, label='v10')
    
    ax.set_title('u10 and v10 at lat={}, lon={}'.format(lat_idx, lon_idx))
    ax.legend()
    
    plt.show()