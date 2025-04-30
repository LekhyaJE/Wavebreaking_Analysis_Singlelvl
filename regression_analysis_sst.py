# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 12:28:45 2025

@author: lenovo
"""
 
""" 
This code carries out LINEAR REGRESSION ANALYSIS of various atmospheric variables (SST,u_wind)
against RWB frequency. 
Input files: RWB at daily frequency, variable 1x1 resolution daily frequency
Output: A spatial map showing regression coeffiecients
Similar Analysis: Takemura 2020 (for the month of August 1958-2018)
The RWB data and wind sata should be on the same time dimension
Either both should be monthly or both should be daily
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
from scipy.stats import linregress
 

#Loading file for RWB events
rwb_data = xr.open_dataset(r"D:\IITB_Project\Data\ERA5\u_winds_1980_2000\merged_streamers_350K.nc")# 40 years annual
rwb_events=rwb_data['stratospheric_streamer']
#Computing Area averaged RWB index  over a target Region
rwb_events = rwb_events.sel(lat=slice(60, 10), lon=slice(55, 105)).mean(dim=['lat', 'lon'])

rwb_events = xr.where(rwb_events > 0, 1, 0)
#Converting daily RWB data to monthly
#rwb_monthly = rwb_events.sel(time=rwb_events['time'].dt.month.isin([8])) # select months
rwb_monthly = rwb_events.resample(time='1M').mean()
rwb_monthly = rwb_monthly[:-1]
area_avg_rwb_index=rwb_monthly


#loading monthly wind data
sst_data=xr.open_dataset(r"D:\IITB_Project\Data\ERA5\SST_1980_2019\sst_monthly_merged_1980_2019.nc")
# Assume u200 is (time, lat, lon), and rwb_index is (time,)
sst_data = sst_data.sel(valid_time=slice("1980-01-31", "2019-11-30"))
sst_kelvin=sst_data['sst']

# Apply conversion to all variables with units in Kelvin
def kelvin_to_celsius(data):
    """
    Convert temperature values from Kelvin to Celsius.
    Works for both xarray.DataArray and xarray.Dataset.
    Only variables with 'K' or 'kelvin' in their units are converted.
    """
    if isinstance(data, xr.DataArray):
        data_celsius = data - 273.15
        data_celsius.attrs['units'] = '°C'
        return data_celsius

    elif isinstance(data, xr.Dataset):
        data_celsius = data.copy()
        for var in data.data_vars:
            if hasattr(data[var], 'units') and ('K' in data[var].units or 'kelvin' in data[var].units.lower()):
                data_celsius[var] = data[var] - 273.15
                data_celsius[var].attrs['units'] = '°C'
        return data_celsius

    else:
        raise TypeError("Input must be an xarray DataArray or Dataset.")


# Usage
sst = kelvin_to_celsius(sst_kelvin)


# Anomaly (optional)
sst_clim = sst.mean(dim='valid_time')
sst_anom = sst - sst_clim  #(time,lat,lon)
sst_anom= sst_anom.rename({'valid_time': 'time'})

rwb_index_anom = rwb_events - area_avg_rwb_index #(time) 1D

sst_anom['time'] = sst_anom.indexes['time'].to_period('M').to_timestamp()
rwb_index_anom['time'] = rwb_index_anom.indexes['time'].to_period('M').to_timestamp()


# Step 1: Align time dimension
common_start = '1980-02-01'
common_end   = '2019-11-01'

sst_anom = sst_anom.sel(time=slice(common_start, common_end))
rwb_index_anom = rwb_index_anom.sel(time=slice(common_start, common_end))

def linregress_wrap(y, x):
    slope, intercept, r, p, stderr = linregress(x, y)
    return slope, p

# Apply along time dimension
regression_slope, regression_p = xr.apply_ufunc(
    linregress_wrap,
    sst_anom,
    rwb_index_anom,
    input_core_dims=[['time'], ['time']],
    output_core_dims=[[], []],
    vectorize=True,
    dask='parallelized',
    output_dtypes=['float32', 'float32']
)

regression_slope = regression_slope/1000


########### PLOTTING #######################
fig = plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=150))
#ax.set_extent([30, 180, 10, 60], crs=ccrs.PlateCarree())
ax.coastlines()

# === Plot Regression Contours ===

# Positive regression (red)
regression_slope.plot.contour(
    ax=ax,
    transform=ccrs.PlateCarree(),
    levels=np.arange(0.2, 4, 0.5),  # tweak as needed,  # tweak as needed
    colors='red',
    linewidths=1
)

# Negative regression (blue)
regression_slope.plot.contour(
    ax=ax,
    transform=ccrs.PlateCarree(),
    levels=np.arange(-1.4, -0.2, 0.2),
    colors='blue',
    linewidths=1
)

# Add hatching only where p < 0.05
# significant_mask = regression_p.where(regression_p < 0.01)
# significant_mask.plot.contourf(
#     ax=ax,
#     transform=ccrs.PlateCarree(),
#     levels=[0, 0.05],  # dummy values to trigger hatching
#     colors='none',
#     hatches=['....'],  # or '///'
#     add_colorbar=False
# )
sig_mask = regression_p < 0.001
sig_mask.plot.contourf(
    ax=ax,
    transform=ccrs.PlateCarree(),
    levels=[0, 0.5, 1],     # binary levels
    hatches=['', '....'],   # hatch only where True
    colors='none',          # transparent fill
    add_colorbar=False
)
import matplotlib.patches as mpatches
patch = mpatches.Patch(facecolor='none', hatch='....', label='p < 0.05', edgecolor='black')
ax.legend(handles=[patch])


# === Aesthetics ===
# Add custom ticks
xticks = np.arange(-180, 181, 30)
yticks = np.arange(-90, 90, 10)

ax.set_xticks(xticks, crs=ccrs.PlateCarree())
ax.set_yticks(yticks, crs=ccrs.PlateCarree())
# Define box coordinates
lon_min, lon_max = 55, 105
lat_min, lat_max = 10, 60

# Create the box as a rectangle
red_box = mpatches.Rectangle(
    (lon_min, lat_min),          # lower-left corner
    lon_max - lon_min,           # width (in degrees)
    lat_max - lat_min,           # height (in degrees)
    transform=ccrs.PlateCarree(), # transform for geographic coords
    linewidth=2,
    edgecolor='red',
    facecolor='none',
    zorder=5                     # plot on top
)

# Add it to the axis0
ax.add_patch(red_box)


ax.set_title('SST Regression onto RWB Frequency', fontsize=13)
plt.show()












