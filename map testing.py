# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 02:43:45 2024

@author: spide
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from folium import plugins
from folium.plugins import HeatMap
import folium

# Generate some example data
longitudes = np.linspace(116.94, 136.94, 21)
# longitudes = df_full['longitude'].to_numpy()
latitudes = np.linspace(16.81, 36.81, 21)
# latitudes = df_full['latitude'].to_numpy()
# LCOH_min, LCOH_max = 4.27, 4.3

# Create a meshgrid for longitude and latitude
lon, lat = np.meshgrid(longitudes, latitudes)

# Generate random LCOH values within the specified range
LCOH = df_full['LCOH'].to_numpy()



df_full['LCOH'] = pd.to_numeric(df_full['LCOH'])


data = df_full.pivot(index='latitude', columns='longitude', values='LCOH')
data_reverse = data.iloc[::-1]


# Plotting the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data_reverse, xticklabels=data_reverse.columns, yticklabels=data_reverse.index, cmap="viridis", cbar_kws={'label': 'LCOH'})

plt.title('Heatmap of LCOH')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()
#%%

avg = df_full['LCOH'].mean()
df_full['LCOH'] = df_full['LCOH'].apply(lambda x: x-avg)      
df_full['LCOH'] = df_full['LCOH'].apply(lambda x: x*100) 

lats = df_full['latitude'].to_numpy()
longs = df_full['longitude'].to_numpy()
LCOH = df_full['LCOH'].to_numpy()

data = {
        "longitude": longs,
        "latitude": lats,
        "LCOH": LCOH}

df = pd.DataFrame(data)

lat_longs = df[['latitude', 'longitude', 'LCOH']].values.tolist()


map_hooray = folium.Map(location=[26.81, 126.94],
                    zoom_start = 3) 

HeatMap(lat_longs, opacity = 0.5, blur = 14).add_to(map_hooray)
# folium.LayerControl().add_to(map_hooray)

# Display the map
map_hooray

map_hooray.save("mymap.html")
