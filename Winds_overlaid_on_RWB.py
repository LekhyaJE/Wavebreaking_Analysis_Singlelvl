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
rwb_data = xr.open_dataset("D:\IITB_Project\Data\merged_daily_overturning_350K.nc")
rwb_events=rwb_data['overturning_event']

flag_data = rwb_events.sel(latitude=slice(60,10), longitude=slice(30, 180))

# Ensure flag is binary (1 for streamer occurrence, 0 otherwise)
flag_data = xr.where(flag_data > 0, 1, 0)

#Loading file for winds
strurl = r"D:\IITB_Project\Data\ERA5\u_winds_1980_2000\uwinds_merged_1980_2000.nc"
demo_data = xr.open_dataset(strurl)
u_winds=demo_data['u'][:,0,:,:]
u_winds = u_winds.sel(latitude=slice(60,10), longitude=slice(30, 180))
lon=demo_data['longitude']
lat=demo_data['latitude']


#Loading ENSO indices
df_enso = pd.read_excel("D:\ENSO_MEI.xlsx")
df_enso=pd.DataFrame(df_enso)
df_long=df_enso.melt(id_vars='YEAR',var_name="MONTH",value_name="ENSO_Index")
# Create correct timestamps
monthly_mapping = {
    'DJ': '-01', 'JF': '-02', 'FM': '-03', 'MA': '-04', 'AM': '-05', 'MJ': '-06',
    'JJ': '-07', 'JA': '-08', 'AS': '-09', 'SO': '-10', 'ON': '-11', 'ND': '-12'}
#creating a new column with pandas datetime64 ns
# ENSO Indices
df_long['month'] = pd.to_datetime(df_long['YEAR'].astype(str) 
                                      + df_long['MONTH'].map(monthly_mapping)).dt.to_period('M')
df_long = df_long.drop(["YEAR","MONTH"], axis=1)

# Convert 'year_month' in both DataFrames to the same string format YYYY-MM
df_long['month'] = df_long['month'].dt.strftime('%Y-%m')


# # Select El Niño months (index >= 0.5)
elnino_df = df_long[df_long['ENSO_Index'] >= 1]
lanina_df = df_long[df_long['ENSO_Index'] <= -1]
neutral_df = df_long[(df_long['ENSO_Index'] > -1) & (df_long['ENSO_Index'] < 1)]


#################### TAKING ENSO COMPOSITES OF WINDS #########################
# Step 1: Ensure elnino_df['date'] is datetime and converted to Period[M]
elnino_df['month'] = pd.to_datetime(elnino_df['month'], format='%Y-%m')
elnino_months = elnino_df['month'].dt.to_period('M')

# Step 1: Ensure elnino_df['date'] is datetime and converted to Period[M]
lanina_df['month'] = pd.to_datetime(lanina_df['month'], format='%Y-%m')
lanina_months = lanina_df['month'].dt.to_period('M')

neutral_df['month'] = pd.to_datetime(neutral_df['month'], format='%Y-%m')
neutral_months = neutral_df['month'].dt.to_period('M')


# Step 3: Convert valid_time in xarray to Period[M] for matching
wind_periods = pd.to_datetime(u_winds.valid_time.values).to_period('M')

# Step 4: Create a boolean mask for El Niño months
elnino_mask = wind_periods.isin(elnino_months)
lanina_mask = wind_periods.isin(lanina_months)
neutral_mask = wind_periods.isin(neutral_months)

# Step 5: Extract El Niño months from u_winds
elnino_u_wind = u_winds.sel(valid_time=u_winds.valid_time[elnino_mask])
lanina_u_wind = u_winds.sel(valid_time=u_winds.valid_time[lanina_mask])
neutral_u_wind = u_winds.sel(valid_time=u_winds.valid_time[neutral_mask])
u_winds_avg = np.mean(u_winds, axis=0)


#Compute Anamolies
elnino_composite_anomaly =  elnino_u_wind - u_winds_avg
lanina_composite_anomaly =  lanina_u_wind - u_winds_avg
neutral_composite_anamoly = neutral_u_wind - u_winds_avg

# Average of anamolies
u_elnino_avg = np.mean(elnino_composite_anomaly, axis=0)
u_lanina_avg = np.mean(lanina_composite_anomaly, axis=0)
u_neutral_avg = np.mean(neutral_composite_anamoly, axis=0)


###################################################################################
################### TAKING ENSO COMPOSITES OF RWB EVENTS ##########################
rwb_periods = pd.to_datetime(flag_data.floor.values).to_period('M')

# Step 4: Create a boolean mask for El Niño months
elnino_mask = rwb_periods.isin(elnino_months)
lanina_mask = rwb_periods.isin(lanina_months)
neutral_mask = rwb_periods.isin(neutral_months)

# Step 5: Extract El Niño months from u_winds
elnino_rwb = flag_data.sel(floor=flag_data.floor[elnino_mask])
lanina_rwb = flag_data.sel(floor=flag_data.floor[lanina_mask])
neutral_rwb = flag_data.sel(floor=flag_data.floor[neutral_mask])

########## PLOTTING ##############################
# List of composite data
uwind_list = [u_elnino_avg, u_lanina_avg, u_neutral_avg]

# Compute spatial frequency maps
freq_elnino = np.sum(elnino_rwb, axis=0)
freq_lanina = np.sum(lanina_rwb, axis=0)
freq_neutral = np.sum(neutral_rwb, axis=0)

# List of data and titles
freq_data = [freq_elnino, freq_lanina, freq_neutral]
titles = [
    "El Niño Composite Anomaly: 200hPa Wind ",
    "La Niña Composite Anomaly: 200hPa Wind ",
    "Neutral Composite Anomaly: 200hPa Wind "
]

# Set contour levels for anamolies
anom_levels = np.arange(-10, 10, 0.5)   # Negative wind speeds

# Define range for contour and colorbar
levels = np.arange(0, 160, 20)  # 0 to 140 (or 160), step of 20

# Get base colormap
base_cmap = plt.get_cmap('tab20c', len(levels) - 1)
# Reverse the colormap
reversed_cmap = ListedColormap(base_cmap.colors[::-1])

# Set 'over' color to white
cmap = base_cmap
cmap.set_over('white')

# Create a norm to handle level boundaries
norm = BoundaryNorm(boundaries=levels, ncolors=cmap.N, clip=False)

############### PLOTTING ##########################
###Here the backgrpund contourfill indicates Rossby wave breaking frequency

# Plotting
fig, axes = plt.subplots(3, 1, figsize=(14, 18), subplot_kw={'projection': ccrs.PlateCarree()})
cf=None
for ax, uwind,rwb,title in zip(axes, uwind_list, 
                                 freq_data, 
                                 titles):
 # --- RWB EVENTS AS FILLED CONTOURS ---

    # Filled contours: POSITIVE values (solid fill)
    cf = ax.contourf(u_winds_avg.longitude, u_winds_avg.latitude, rwb.values,
                         levels=levels, 
                         cmap=reversed_cmap,
                         norm=norm, 
                         extend='max',
                         transform=ccrs.PlateCarree(),alpha=0.5)

    # --- OVERLAY COMPOSITE ANOMALIES (LINE CONTOURS) ---

    # Negative anomaly contours
    neg_anom = ax.contour(uwind.longitude, uwind.latitude, uwind.values,
                          levels=[lvl for lvl in anom_levels if lvl < 0],
                          colors='blue', linestyles='dotted', linewidths=1,
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
    ax.set_extent([30, 120, 10, 60], crs=ccrs.PlateCarree())
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
cbar.set_label('ENSO composite 200hPa zonal winds overlaid on RWB events', fontsize=16, weight='bold')
cbar.ax.tick_params(labelsize=16)
cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))

plt.subplots_adjust(right=1.2, hspace=0.3)
plt.show()






