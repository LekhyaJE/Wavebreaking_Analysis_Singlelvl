# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 15:21:28 2025

@author: lenovo
"""

#This code considers WAVE BREAKING events only in case of
#co exitence of both stratopsheric and tropospheric events


import xarray as xr
import numpy as np
import pandas as pd
from scipy.ndimage import label, center_of_mass
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import calendar
import seaborn as sns


# Load dataset
strurl = r"D:\merged_streamers_350K_1PV.nc"
demo_data = xr.open_dataset(strurl)
strato_flag_data = demo_data['stratospheric_streamer']
tropo_flag_data = demo_data['tropospheric_streamer']


strato_clipped_data = strato_flag_data.sel(latitude=slice(40,0), longitude=slice(55, 105))
tropo_clipped_data = tropo_flag_data.sel(latitude=slice(40,0), longitude=slice(55, 105))


# Convert to binary (1 for presence, 0 otherwise)
tropo_streamers = xr.where(tropo_clipped_data > 0, 1, 0)
strato_streamers = xr.where(strato_clipped_data > 0, 1, 0)



def count_streamer_events(tropo_streamers, strato_streamers):
    total_events = 0
    
    # Loop over time
    for t in range(tropo_streamers.shape[0]):  
        # Label connected components in each layer
        tropo_labels, num_tropo = label(tropo_streamers[t])  # Tropospheric streamers
        strato_labels, num_strato = label(strato_streamers[t])  # Stratospheric streamers

        if num_tropo == 0 or num_strato == 0:
            continue  # Skip if no streamers are present

        # Compute centroids of labeled streamers
        tropo_centroids = np.array(center_of_mass(tropo_streamers[t], tropo_labels, np.arange(1, num_tropo + 1)))
        strato_centroids = np.array(center_of_mass(strato_streamers[t], strato_labels, np.arange(1, num_strato + 1)))

        if len(tropo_centroids) == 0 or len(strato_centroids) == 0:
            continue  # Skip if centroids cannot be determined

        # Use KDTree to find the nearest tropospheric streamer for each stratospheric streamer
        tree = cKDTree(tropo_centroids)
        _, nearest_tropo_indices = tree.query(strato_centroids)

        # Count unique tropospheric streamers that have at least one connection
        unique_connected_tropo_streamers = np.unique(nearest_tropo_indices)
        total_events += len(unique_connected_tropo_streamers)
    
    return total_events

# Example usage:
# tropo_streamers and strato_streamers are 3D arrays (time, latitude, longitude) with 1s where streamers exist and 0s elsewhere
num_events = count_streamer_events(tropo_streamers, strato_streamers)
print(f"Total streamer connection events: {num_events}")



# Group by month and count occurrences
time_index = demo_data.floor  # Assuming 'time' is the time coordinate
monthly_counts = co_occurrence.groupby(time_index.dt.month).sum().values  # Sum occurrences per month


# Convert to Pandas Series for easier plotting
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
monthly_series = pd.Series(monthly_counts, index=months)

# Plot the bar chart
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(months, monthly_series, color='purple', edgecolor='black', alpha=0.8)

# Formatting
ax.set_xlabel("Month", fontsize=14, fontweight='bold')
ax.set_ylabel("Number of Co-occurrence Events", fontsize=14, fontweight='bold')
ax.set_title("Monthly Distribution of RWB events", fontsize=16, fontweight='bold')
ax.tick_params(axis='both', labelsize=12)

plt.xticks(rotation=45, fontsize=12, weight='bold')  # Rotate month labels
plt.yticks(fontsize=12, weight='bold')  # Set y-tick font size

plt.show()
