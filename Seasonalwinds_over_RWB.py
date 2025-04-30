
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 11:37:50 2025

@author: lenovo
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
flag_data = xr.where(rwb_events > 0, 1, 0)


rwb_jjas = flag_data.sel(time=rwb_events['time'].dt.month.isin([6,7,8,9])) #JJAS
rwb_jjas = np.sum(rwb_jjas,axis=0)
rwb_djf = flag_data.sel(time=rwb_events['time'].dt.month.isin([1,2,12])) #DJF
rwb_djf = np.sum(rwb_djf,axis=0)


#Loading file for winds
strurl = r"D:\IITB_Project\Data\ERA5\u_winds_1980_2000\uwinds_merged_1980_2019.nc"#daily frequency/could give monthly also
# Subset the time range
demo_data = xr.open_dataset(strurl)
u_winds=demo_data['u'][:,0,:,:]
u_winds_djf = u_winds.sel(valid_time=u_winds['valid_time'].dt.month.isin([1,2,12]))
u_winds_jjas =  u_winds.sel(valid_time=u_winds['valid_time'].dt.month.isin([6,7,8,9])) 


lon=demo_data['longitude']
lat=demo_data['latitude']

u_winds_avg = np.mean(u_winds, axis=0)
uwinds_djf_anomaly = u_winds_djf - u_winds_avg
uwinds_jjas_anomaly = u_winds_jjas - u_winds_avg

djf_anom_winds = np.mean(uwinds_djf_anomaly, axis=0)
jjas_anom_winds = np.mean(uwinds_jjas_anomaly, axis=0) #inka countour plot banaani hey



########## PLOTTING ##############################
# Set contour levels for anamolies
anom_levels = np.arange(-30, 35, 10)   # (a,b,c) c is the gap between the contour levels

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



# --- RWB EVENTS AS FILLED CONTOURS ---
# Mask for positive and negative climatology values
# clim_data = u_winds_avg.values
# pos_mask_djf = np.ma.masked_less_equal(djf_anom_winds, 0) #masks (hides/ignores) all
# pos_mask_jjas = np.ma.masked_less_equal(jjas_anom_winds, 0)#values that are less than or equal to 0.

# neg_mask_djf = np.ma.masked_greater_equal(djf_anom_winds, 0)
# neg_mask_jjas = np.ma.masked_greater_equal(jjas_anom_winds, 0)


############### PLOTTING ##########################
###Here the backgrpund contourfill indicates Rossby wave breaking frequency
rwb_data = [rwb_jjas,rwb_djf]
uwind_list = [jjas_anom_winds, djf_anom_winds]
titles = ['JJAS', 'DJF']

# Plotting
fig, axes = plt.subplots(2, 1, figsize=(14, 20), subplot_kw={'projection': ccrs.EckertIV()})
cf=None
for ax, uwind,rwb,title in zip(axes, 
                               uwind_list, 
                               rwb_data, 
                               titles):
 # --- RWB EVENTS AS FILLED CONTOURS ---

    cf = ax.contourf(rwb.lon, rwb.lat, rwb.values,
                         levels=levels, 
                         cmap=reversed_cmap,
                         norm=norm, 
                         extend='max',
                         transform=ccrs.PlateCarree(),alpha=0.5)

    # --- OVERLAY WIND ANOMALIES (LINE CONTOURS) ---

    # Negative anomaly contours
    neg_anom = ax.contour(uwind.longitude, uwind.latitude, uwind.values,
                          levels=[lvl for lvl in anom_levels if lvl < 0],
                          colors='blue', linestyles='dotted', linewidths=2,
                          transform=ccrs.PlateCarree())

    # Positive anomaly contours
    pos_anom = ax.contour(uwind.longitude, uwind.latitude, uwind.values,
                          levels=[lvl for lvl in anom_levels if lvl > 0],
                          colors='red', linestyles='solid', linewidths=1,
                          transform=ccrs.PlateCarree())

    # Add contour labels
    ax.clabel(neg_anom, inline=True, fontsize=10, fmt='%d', colors='blue')
    ax.clabel(pos_anom, inline=True, fontsize=10, fmt='%d', colors='red')

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



# --- COLORBAR FOR CLIMATOLOGY FILL ---
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
cbar = fig.colorbar(cf, cax=cbar_ax, orientation='vertical')
cbar.set_label('RWB frequency', fontsize=16, weight='bold')
cbar.ax.tick_params(labelsize=16)
cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))

plt.subplots_adjust(right=0.9, hspace=0.3)
plt.show()






