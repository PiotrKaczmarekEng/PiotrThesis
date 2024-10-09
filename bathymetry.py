# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 14:26:33 2024

@author: spide
"""

import xarray as xr
import numpy as np
import os


def get_water_depth(lat, lon, bathymetry_file='NetCDF/north_sea_bathymetry.nc'):
    """
    Returns the water depth (bathymetry) for a given latitude and longitude from a bathymetry dataset.

    Parameters:
    lat (float): Latitude of the location
    lon (float): Longitude of the location
    bathymetry_file (str): Path to the NetCDF bathymetry file

    Returns:
    float: Water depth at the specified lat/lon location in meters. Positive values indicate depth.
    """

    # Open the NetCDF file containing bathymetry data
    try:
        ds = xr.open_dataset(bathymetry_file)
    except FileNotFoundError:
        print("Bathymetry file not found.")
        return None

    # Check if the file contains 'latitude', 'longitude', and 'elevation' variables
    if 'lat' not in ds.variables or 'lon' not in ds.variables or 'elevation' not in ds.variables:
        print("NetCDF file does not contain required variables.")
        return None

    # Find the nearest lat/lon indices in the dataset
    lat_idx = np.abs(ds['lat'] - lat).argmin().item()
    lon_idx = np.abs(ds['lon'] - lon).argmin().item()

    # Extract the water depth at the nearest lat/lon indices
    water_depth = ds['elevation'][lat_idx, lon_idx].item()

    # Close the dataset
    ds.close()

    return water_depth

# Example usage:
lat = 58.35  # Example latitude
lon = 2.78   # Example longitude
depth = get_water_depth(lat, lon)
print(f"Water depth at lat: {lat}, lon: {lon} is {depth} meters.")



# Optimal j=1,2:
lat = 57.35,   # Example latitude
lon = 7.78   # Example longitude
depth = get_water_depth(lat, lon)
print(f"Water depth at lat: {lat}, lon: {lon} is {depth} meters.")

# Optimal j=3,4:
lat = 54.35   # Example latitude
lon = 6.78   # Example longitude
depth = get_water_depth(lat, lon)
print(f"Water depth at lat: {lat}, lon: {lon} is {depth} meters.")

size = 6
for i in range(size):
    for j in range(size):
        lat=[loc_matrix[i,j][1]]
        lon=[loc_matrix[i,j][0]]
        depth = get_water_depth(lat, lon)
        print(f"Water depth at lat: {lat}, lon: {lon} is {depth} meters.")
        
        
        
#%%

df_map_depth = pd.DataFrame(columns=['longitude','latitude','depth'],index=[list(range(size*size))])

counter = 0
for i in range(size):
    for j in range(size):
        lat=loc_matrix[i,j][1]
        lon=loc_matrix[i,j][0]
        depth = get_water_depth(lat, lon)*-1
        df_map_depth.loc[counter] = [lon, lat, depth]
        counter += 1



df_map_depth = df_map_depth.loc[(df_map_depth['depth'] >0)]

fig_map = px.density_mapbox(df_map_depth, lat = 'latitude', lon = 'longitude', z = 'depth',
                        radius = 30,
                        center = dict(lat = latitude, lon = longitude),
                        zoom = 3,
                        mapbox_style = 'open-street-map',
                        color_continuous_scale = 'rainbow')


# Adjust color of heatmap by adding more points for density
fig_map.add_trace(
    go.Scattermapbox(
        lat=df_map_depth["latitude"],
        lon=df_map_depth["longitude"],
        mode="markers",
        showlegend=False,
        hoverinfo="skip",
        marker={
            "color": df_map_depth["depth"],
            "size": df_map_depth["depth"].fillna(0).infer_objects(copy=False),
            "coloraxis": "coloraxis",
            # desired max size is 15. see https://plotly.com/python/bubble-maps/#united-states-bubble-map
            "sizeref": (df_map_depth["depth"].max()) / 30 ** 2,
            "sizemode": "area",
        },
    )
)


pio.renderers.default='browser'
fig_map.show()

fig_map.data = []