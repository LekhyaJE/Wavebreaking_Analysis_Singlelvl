# -*- coding: utf-8 -*-
"""
Created on Tue May 27 09:59:01 2025

@author: lenovo

Annual and decadal frequency of RWB events over India
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

# === Load RWB event data ===
rwb_data = xr.open_dataset(r"D:\IITB_Project\Data\rwb_greaterthan_1lvl.nc")
rwb_events = rwb_data['overturning_event']

# Ensure flag is binary (1 for streamer occurrence, 0 otherwise)
flag_data = xr.where(rwb_events > 0, 1, 0)

rwb_events = flag_data.sel(lat=slice(40,0), lon=slice(55, 105))



# === STEP 1: Identify if RWB exists per day (anywhere in domain) ===
times = rwb_events.time.values
event_present = []

for i, time in enumerate(times):
    data = rwb_events.isel(time=i).values
    _, num_features = label(data)
    event_present.append(1 if num_features > 0 else 0)

# Convert to pandas Series
event_series = pd.Series(event_present, index=pd.to_datetime(times))

# === STEP 2: Label continuous RWB sequences and filter ≥3 day events ===
event_array = event_series.values
event_labels, num_events = label(event_array)

valid_event_start_dates = []

for label_id in range(1, num_events + 1):
    indices = np.where(event_labels == label_id)[0]
    if len(indices) >= 3:
        valid_event_start_dates.append(event_series.index[indices[0]])  # Record only the start date

# === STEP 3: Count valid events per year ===
event_years = pd.Series(valid_event_start_dates).dt.year
annual_event_counts = event_years.value_counts().sort_index()


# === STEP 4: Create DataFrame for plotting ===
rwb_df = pd.DataFrame({
    'year': annual_event_counts.index,
    'overturning_event': annual_event_counts.values
})
rwb_df['running_mean'] = rwb_df['overturning_event'].rolling(window=10, center=True).mean()


# === ENSO Phase Classification (1980–2019) ===
elnino_years = [1982, 1983, 1987, 1988, 1991, 1992, 1994, 1995, 1997, 1998,
                2002, 2003, 2004, 2005, 2006, 2007, 2009, 2010, 2014, 2015, 2018, 2019]
lanina_years = [1984, 1985, 1989, 1990, 1995, 1996, 1999, 2000, 2007, 2008,
                2010, 2011, 2016, 2017]

# Neutral years = everything else from 1980 to 2019
all_years = list(range(1980, 2020))
neutral_years = [yr for yr in all_years if yr not in elnino_years and yr not in lanina_years]


# === Assign Colors by ENSO Phase ===
def assign_color(year):
    if year in elnino_years:
        return 'red'
    elif year in lanina_years:
        return 'blue'
    else:
        return 'grey'

rwb_df['color'] = rwb_df['year'].apply(assign_color)



# Plotting
# === Plotting ===
plt.figure(figsize=(12, 8))
plt.bar(
    rwb_df['year'], 
    rwb_df['overturning_event'], 
    width=0.8, 
    color=rwb_df['color']
)

plt.plot(
    rwb_df['year'], 
    rwb_df['running_mean'], 
    color='black', 
    linewidth=2, 
    marker='*', 
    markersize=10,
    label='10-Year Running Mean'
)

plt.xlabel("Year", fontsize=12, fontweight='bold')
plt.ylabel("Number of RWB events", fontsize=12, fontweight='bold')
plt.title("Annual Frequency of RWB Events over India", fontsize=14, fontweight='bold')
plt.xticks(ticks=rwb_df['year'][::5], rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)


# Custom Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='red', label='El Niño'),
    Patch(facecolor='blue', label='La Niña'),
    Patch(facecolor='grey', label='Neutral'),
    plt.Line2D([0], [0], color='black', lw=2, marker='*', markersize=10, label='10-Year Running Mean')
]
plt.legend(handles=legend_elements)

plt.tight_layout()
plt.savefig(r"D:\IITB_Project\Plots\annual_rwb_climatology_enso.png", dpi=300)
plt.show()



