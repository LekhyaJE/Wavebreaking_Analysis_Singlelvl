# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 12:16:04 2025

@author: lenovo

I'm also implementing a GIF in this
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
import os
import imageio

##### ts. 

#Loading file for RWB events
rwb_data = xr.open_dataset(r"D:\IITB_Project\Data\ERA5\u_winds_1980_2000\merged_streamers_350K.nc")
rwb_events=rwb_data['stratospheric_streamer']

lat=rwb_events['lat']
lon=rwb_events['lon']

# Create meshgrid for plotting
lon2d, lat2d = np.meshgrid(lon, lat)
# Select only June, July, and August
flag_jja = rwb_events.sel(time=rwb_events['time'].dt.month.isin([1,2,12]))

# Ensure flag is binary (1 for streamer occurrence, 0 otherwise)
flag_data = xr.where(flag_jja > 0, 1, 0)

rwb_yearly = flag_data.resample(time='1Y').sum()
 # shape: (lat, lon)

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

# === Prepare output directory for GIF frames ===
output_dir = "rwb_gif_frames"
os.makedirs(output_dir, exist_ok=True)

# === Loop through each year to generate a frame ===
image_files = []
# === Determine max RWB value across all years for consistent colorbar ===
rwb_max=np.max(rwb_yearly.values)
levels = np.linspace(0, rwb_max, 12)  # consistent levels
norm = BoundaryNorm(levels, ncolors=len(colors), clip=True)

for i in range(len(rwb_yearly.time)):
 data = rwb_yearly.isel(time=i)
 year = pd.to_datetime(str(data.time.values)).year
# Set up the polar stereographic plot
 fig = plt.figure(figsize=(8, 8))
 ax = plt.subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())

# Set extent: Northern Hemisphere only
 ax.set_extent([-180, 180, 30, 90], crs=ccrs.PlateCarree())
 ax.coastlines()
 ax.gridlines(draw_labels=False, color='gray', alpha=0.5, linestyle='--')
 n_colors = len(colors)
 custom_cmap = LinearSegmentedColormap.from_list("custom_rwb", colors, N=n_colors)
 cs = ax.contourf(lon2d, lat2d, data, levels=levels, cmap=custom_cmap, transform=ccrs.PlateCarree(), extend='both')

# Add colorbar
 cbar = plt.colorbar(cs, orientation='horizontal', pad=0.05, shrink=0.8)
 cbar.set_label("RWB Frequency")
 plt.title(f"Annual Rossby Wave Breaking Events,Year:{year} ")
 plt.tight_layout()
 
 
 # Save frame
 filename = os.path.join(output_dir, f"rwb_{year}.png")
 plt.savefig(filename, dpi=150)
 image_files.append(filename)
 plt.close()
# === Create GIF ===
gif_path = "D:\IITB_Project\Plots\RWB_yearly_DJF.gif"
with imageio.get_writer(gif_path, mode='I', duration=0.7) as writer:
    for filename in image_files:
        image = imageio.imread(filename)
        writer.append_data(image)

print(f"GIF saved as {gif_path}")
