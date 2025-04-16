# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 13:25:38 2025

@author: lenovo

This code lets you plot the longitudinal distribution of Rossby breaking events along a fixed latitude band
Input file: The .nc file you get after running the Wavebreaking algorithm (Temporal resolution: Daily, Spatial resolution: 1 degree)
Output: A decently goodlooking line plot with RWB occurence fequency on y-axis, longitudes on x-axis
        Red line: indicates the rossby wave breaking events during Elnino period (El Nino composite of RWB events)
        Blue Line: indicates the rossby wave breaking events during LaNina period (La nina composite of RWB events)
        Grey Line: indicates the rossby wave breaking events during Neutral ENSO period (Neutral composite of RWB events)
"""

#### Longitudinal distribution ENSO composite RWB events ############
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
##### This code shows ENSO composite longitudinal RWB freq

#Loading file for RWB events
streamers = xr.open_dataset("D:\IITB_Project\Data\merged_streamers_350K_2PV.nc")
streamers=streamers['stratospheric_streamer']

cutoffs = xr.open_dataset("D:\IITB_Project\Data\merged_cutoffs_350K_2PV.nc")
cutoffs=cutoffs['stratospheric_cutoff']


streamer_flag_data = streamers.sel(latitude=slice(50,0))
cutoff_flag_data = cutoffs.sel(latitude=slice(50,0))

# Ensure flag is binary (1 for streamer occurrence, 0 otherwise)
streamer_flag_data = xr.where(streamer_flag_data > 0, 1, 0)
cutoff_flag_data = xr.where(cutoff_flag_data > 0, 1, 0)

flag_data = xr.where((streamer_flag_data | cutoff_flag_data) > 0, 1, 0)

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

# Step 1: Ensure elnino_df['date'] is datetime and converted to Period[M]
elnino_df['month'] = pd.to_datetime(elnino_df['month'], format='%Y-%m')
elnino_months = elnino_df['month'].dt.to_period('M')

# Step 1: Ensure elnino_df['date'] is datetime and converted to Period[M]
lanina_df['month'] = pd.to_datetime(lanina_df['month'], format='%Y-%m')
lanina_months = lanina_df['month'].dt.to_period('M')

neutral_df['month'] = pd.to_datetime(neutral_df['month'], format='%Y-%m')
neutral_months = neutral_df['month'].dt.to_period('M')


rwb_periods = pd.to_datetime(flag_data.floor.values).to_period('M')

# Step 4: Create a boolean mask for El Niño months
elnino_mask = rwb_periods.isin(elnino_months)
lanina_mask = rwb_periods.isin(lanina_months)
neutral_mask = rwb_periods.isin(neutral_months)

# Step 5: Extract El Niño months from u_winds
elnino_rwb = flag_data.sel(floor=flag_data.floor[elnino_mask])
lanina_rwb = flag_data.sel(floor=flag_data.floor[lanina_mask])
neutral_rwb = flag_data.sel(floor=flag_data.floor[neutral_mask])

# Compute spatial frequency maps
freq_elnino = np.sum(elnino_rwb, axis=0)
freq_lanina = np.sum(lanina_rwb, axis=0)
freq_neutral = np.sum(neutral_rwb, axis=0)

longi_elnino = np.sum(freq_elnino, axis=0)
longi_lanina = np.sum(freq_lanina, axis=0)
longi_neutral = np.sum(freq_neutral, axis=0)

longitudes = flag_data.longitude.values

totals=longi_elnino+longi_lanina+longi_neutral

elnino_pct=[]
lanina_pct=[]
neutral_pct=[]
# Convert to percentages
for i in range(360):
    en_pct = 100 *(longi_elnino[i] / np.sum(totals,axis=0))
    elnino_pct.append(en_pct)
    ln_pct = 100 *(longi_lanina[i] /np.sum(totals,axis=0))
    lanina_pct.append(ln_pct)
    nt_pct = 100 *(longi_neutral[i] /np.sum(totals,axis=0))
    neutral_pct.append(nt_pct)
    
plt.figure(figsize=(10, 6))
plt.plot(longitudes, elnino_pct, label='El Niño', color='red', linewidth=4)
plt.plot(longitudes, lanina_pct, label='La Niña', color='blue', linewidth=4)
plt.plot(longitudes, neutral_pct, label='Neutral', color='gray', linewidth=4)

plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(1.0))  # <- THIS
plt.ylabel("Percentage of RWB Events",fontsize='14')
plt.title('ENSO composite of RWB longitudinal distribution', fontsize='14')
plt.xticks(np.arange(0, 361, 30))
# Call legend explicitly
plt.legend(loc='upper right')  # or try 'best', 'upper left', etc.
plt.tight_layout()
plt.show()









