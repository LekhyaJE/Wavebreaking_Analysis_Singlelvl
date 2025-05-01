# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 13:35:55 2025

@author: lenovo

This code is for Wind Composite Analysis of RWB events.
How the RWB events are distributed wrt the wind speeds of jets
How the RWB events are placed when winds are strong (top 25 percentile)
How the RWB events are placed when winds are weak (bottom 25 percentile)

"""
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import label, gaussian_filter
import calendar
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import ListedColormap, BoundaryNorm

##### This code shows 200hPa ENSO composite winds overlaid on RWB events. 

#Loading file for RWB events
rwb_data = xr.open_dataset(r"D:\IITB_Project\Data\ERA5\u_winds_1980_2000\merged_streamers_350K.nc")#3_5_conservative_output_dailyFrequency

rwb_events=rwb_data['stratospheric_streamer']

#flag_data = rwb_events.sel(latitude=slice(60,10), longitude=slice(30, 180))

# Ensure flag is binary (1 for streamer occurrence, 0 otherwise)
flag_data = xr.where(rwb_events > 0, 1, 0) #also 3d (daily)
rwb_monthly = flag_data.resample(time='1M').sum() #It sums over each calendar month, based on the actual dates in the time coordinate.

#Loading file for winds
strurl = r"D:\IITB_Project\Data\ERA5\u_winds_1980_2000\uwinds_merged_1980_2019.nc"#daily frequency/could give monthly also
# Subset the time range
demo_data = xr.open_dataset(strurl)
u_winds=demo_data['u'][:,0,:,:] # 3d (monthly)

lon=demo_data['longitude']
lat=demo_data['latitude']


# Aligning RWB and wind data in time
u_winds['valid_time'] = u_winds.indexes['valid_time'].to_period('M').to_timestamp()
rwb_monthly['time'] = rwb_monthly.indexes['time'].to_period('M').to_timestamp()


#Calculate  Jet index
jet_index = u_winds.mean(dim=['latitude', 'longitude'])
#Jet_index is now a time series that tracks how strong the jet is on each day (or time step).

#Using absolute values of Jet speed becoz Strong easterlies might be considered 
#under bottom percentile and classified weaker jets which isnt the case
jet_index_abs = np.abs(jet_index)

# Get Top 25 percentile and bottom 25 percentile of winds
low_thresh = jet_index.quantile(0.25, dim='valid_time')
high_thresh = jet_index.quantile(0.75, dim='valid_time')

# Get dates for strong and weak jet cases
weak_jet_days = jet_index.valid_time[jet_index < low_thresh]
strong_jet_days = jet_index.valid_time[jet_index > high_thresh]

#Get RWB composites on weak jet days and strong jet days
rwb_weak_jet =  rwb_monthly.sel(time=weak_jet_days)
rwb_weak_jet = np.sum(rwb_weak_jet,axis=0)

rwb_strong_jet = rwb_monthly.sel(time=strong_jet_days)
rwb_strong_jet = np.sum(rwb_strong_jet,axis=0)


# Plotting#############################################
# Define range for contour and colorbar
levels = np.arange(0, 700, 40)  # 0 to 140 (or 160), step of 20

# Get base colormap
base_cmap = plt.get_cmap('tab20c', len(levels) - 1)
# Reverse the colormap
reversed_cmap = ListedColormap(base_cmap.colors[::-1])

# Set 'over' color to white
cmap = base_cmap
cmap.set_over('white')

# Create a norm to handle level boundaries
norm = BoundaryNorm(boundaries=levels, ncolors=cmap.N, clip=False)

jets=[rwb_weak_jet,rwb_strong_jet]

titles = ['RWB when the jet is weak','RWB when the jet stream is strong']

fig, axes = plt.subplots(2, 1, figsize=(14, 20), subplot_kw={'projection': ccrs.EckertIV(central_longitude=150)})
cf = None
for ax, data, title in zip(axes,jets,titles):
   
# --- RWB EVENTS AS FILLED CONTOURS ---
   cf = ax.contourf(data.lon, data.lat, data.values,
                        levels=levels, 
                        cmap=reversed_cmap,
                        norm=norm, 
                        extend='max',
                        transform=ccrs.PlateCarree(),alpha=0.5)
   # --- MAP FEATURES ---
   #ax.set_extent([30, 120, 10, 60], crs=ccrs.PlateCarree())
   ax.coastlines()
   ax.add_feature(cfeature.BORDERS, linestyle=':')
   ax.set_title(title, fontsize=16, weight='bold')

   # Axis font styling
   for tick in ax.get_xticklabels() + ax.get_yticklabels():
       tick.set_fontsize(16)
       tick.set_fontweight('bold')

   # Gridlines
   gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.7)
   gl.right_labels = False
   gl.top_labels = False

# Colorbar (shared)
cbar = fig.colorbar(cf, ax=axes, orientation='vertical', fraction=0.05, pad=0.01)
cbar.set_label('RWB Frequency')

plt.show()




