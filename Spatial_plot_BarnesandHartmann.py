# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 21:48:03 2025

@author: lenovo
"""

"""
Global distribution of RWB evevnts as in Barnes and Hartmann(2012)
Each pixel: Rossby wave breaking frequency peryear per degree latitude

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
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from matplotlib import cm

##### This code shows 200hPa ENSO composite winds overlaid on RWB events. 

#Loading file for RWB events
rwb_data = xr.open_dataset("D:\IITB_Project\Data\stratospheric_streamers_350K.nc")
rwb_events=rwb_data['flag']

# Select only June, July, and August
flag_jja = rwb_events.sel(time=rwb_events['time'].dt.month.isin([6, 7, 8]))


# Ensure flag is binary (1 for streamer occurrence, 0 otherwise)
flag_data = xr.where(flag_jja > 0, 1, 0)


# Convert time to Pandas datetime for grouping
flag_data['time'] = pd.to_datetime(flag_data['time'].values)


# Step 1: Count RWB events per grid cell
rwb_count = flag_data.mean(dim='time')  # shape: (lat, lon)


# Define the custom colormap: white → grey → blue → yellow → orange → red
colors = [
    "#ffffff",  # white
    "#e6e6e6",  # very light grey
    "#cccccc",  # light grey
    "#a3cce6",  # soft blue-grey
    "#7fbfff",  # light blue
    "#b2d9ff",  # pale blue
    "#ffffbf",  # light yellow
    "#ffe090",  # yellow-orange
    "#fdae61",  # orange
    "#f46d43",  # strong orange
    "#d73027",  # red
    "#a50026"   # dark red
]

# Create the colormap with as many levels as needed
n_colors = len(colors)
custom_cmap = LinearSegmentedColormap.from_list("custom_rwb", colors, N=n_colors)

# Define levels based on your data
levels = np.arange(np.min(rwb_count),np.max(rwb_count), 0.0125)  # For example, matching your colorbar 0 to 0.2

# Normalize based on levels
norm = BoundaryNorm(levels, ncolors=custom_cmap.N, clip=True)


# Step 4: Plot
fig = plt.figure(figsize=(8, 5))
# Shift central longitude to 150°W (or -210°)
proj = ccrs.PlateCarree(central_longitude=150)
ax = plt.axes(projection=proj)

# Pixellated map
mesh = ax.pcolormesh(
    flag_data.lon, flag_data.lat, rwb_count,
    cmap=custom_cmap,
    alpha=0.8,
    norm=norm,
    shading='nearest',  # ensures a pixelated look
    transform=ccrs.PlateCarree()
)

# Add map features
ax.coastlines()
ax.set_global()
ax.add_feature(cfeature.BORDERS, linewidth=0.3)
ax.set_title(r'RWB Frequency ($\mathrm{year^{-1} \, deg^{-1} \, lat \, deg^{-1} \, lon}$)', fontsize=12)
ax.set_extent([-170, 180, -90, 90], crs=ccrs.PlateCarree())
# Add colorbar
cbar = plt.colorbar(mesh, orientation='vertical', shrink=0.75, pad=0.02)
cbar.set_label("WB frequency\n(year⁻¹ deg⁻¹ lat deg⁻¹ lon)", fontsize=10)
cbar.set_ticks(levels)

plt.tight_layout()
plt.show()


