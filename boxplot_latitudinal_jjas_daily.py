# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 19:04:55 2025

@author: lenovo
"""

import xarray as xr
import numpy as np
import pandas as pd
from scipy.ndimage import label
import matplotlib.pyplot as plt
import calendar
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# Load dataset
strurl = r"D:\IITB_Project\Data\rwb_greaterthan_1lvl.nc"
demo_data = xr.open_dataset(strurl)
demo_data = demo_data['overturning_event']
flag_data = demo_data.sel(lat = slice(40,0), lon=slice(55,105))

# Ensure flag is binary (1 for streamer occurrence, 0 otherwise)
flag_data = xr.where(flag_data > 0, 1, 0)

latitudes = flag_data['lat']  # Assuming latitude is a coordinate
longitudes = flag_data['lon']  # Assuming latitude is a coordinate


# Initialize lists to store counts and lowest latitude per timestep
daily_counts = []
lowest_latitudes = []
times = flag_data.time.values  # Assuming 'time' is a coordinate

# Create 2D latitude array from coordinates
lat2d = flag_data['lat'].broadcast_like(flag_data.isel(time=0)).values

# Loop through each time step
for i, time in enumerate(times):   
    binary_data = flag_data.isel(time=i).values.astype(bool)
    labeled_array, num_features = label(binary_data)
    
    daily_counts.append((time, num_features))
    lowest_lat = None
   
    for label_id in range(1, num_features + 1):
        # Boolean mask of current streamer
        mask = labeled_array == label_id

        # Use lat2d and mask together
        streamer_latitudes = lat2d[mask]

        if streamer_latitudes.size > 0:
            min_lat = np.nanmin(streamer_latitudes)
            if lowest_lat is None or min_lat < lowest_lat:
                lowest_lat = min_lat
    
    if lowest_lat is None:
        lowest_lat = np.nan
    
    lowest_latitudes.append((time, lowest_lat))
    
    
df = pd.DataFrame(lowest_latitudes, columns=['time', 'lowest_latitude'])

df['time'] = pd.to_datetime(df['time'])

# Filter for JJAS season (June 1 to September 30)
df = df[(df['time'].dt.month >= 6) & (df['time'].dt.month <= 9)]

# Create a 'day' column as MM-DD to group across years
df['day'] = df['time'].dt.strftime('%m-%d')

# Group by day-of-season (e.g., all June 01 across years)
daily_grouped = df.groupby('day')['lowest_latitude'].apply(list).reset_index()

# Sort by actual calendar day
daily_grouped['day_sort'] = pd.to_datetime("2000-" + daily_grouped['day'])  # Dummy year to sort
daily_grouped = daily_grouped.sort_values('day_sort').reset_index(drop=True)

# Explode for boxplot
daily_grouped = daily_grouped.explode('lowest_latitude')
daily_grouped['lowest_latitude'] = pd.to_numeric(daily_grouped['lowest_latitude'], errors='coerce')

# Plot boxplot for all 122 JJAS days
plt.figure(figsize=(28, 6))

sns.boxplot(
    x='day', 
    y='lowest_latitude', 
    data=daily_grouped, 
    color='lightblue',
    showmeans=True,
    meanprops={"marker":"o", "markerfacecolor":"black", "markeredgecolor":"black", "markersize":6},
    boxprops=dict(edgecolor="black"),
    medianprops=dict(color="black", linewidth=1.5),
    whiskerprops=dict(color="black", linestyle="-"),
    capprops=dict(color="black"),
    flierprops=dict(marker="o", color="green", alpha=0.4)
)

plt.title("Daily RWB Latitudinal Extent (JJAS, 1980–2019)", fontsize=20, weight='bold')
plt.ylabel("Latitude (N)", fontsize=16, weight='bold')
plt.xlabel("Day of JJAS Season", fontsize=16, weight='bold')
plt.xticks([1, 31, 62, 93], ['Day 1', 'Day 31', 'Day 62', 'Day 93'],rotation=90, fontsize=12)
#plt.xticks(rotation=90, fontsize=8)
plt.yticks(fontsize=12)
plt.tight_layout()

# Optional: Save the figure
plt.savefig(r"D:\IITB_Project\Plots\Multiple_level_analysis\latitudinal_intrusions_JJAS_daily_boxplot.png", dpi=300)
plt.show()