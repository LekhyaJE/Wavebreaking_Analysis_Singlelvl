# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 22:14:29 2025

@author: lenovo

Not applying 
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

# Convert to DataFrame for monthly statistics
df = pd.DataFrame(lowest_latitudes, columns=['time', 'lowest_latitude'])
df['time'] = pd.to_datetime(df['time'])
df['month'] = df['time'].dt.to_period('M')
# Convert 'month' to month name (removes year)
df["month"] = df["month"].dt.strftime("%B")  # Converts '2001-01' → 'January'


# Group by month and collect all latitude values in a list
monthly_grouped = df.groupby("month")["lowest_latitude"].apply(list).reset_index()

# Define the correct month order
month_order = ["January", "February", "March", "April", "May", "June", 
               "July", "August", "September", "October", "November", "December"]


# Sort DataFrame based on month order
monthly_grouped["month"] = pd.Categorical(monthly_grouped["month"], categories=month_order, ordered=True)
monthly_grouped = monthly_grouped.sort_values("month").reset_index(drop=True)


# Apply function to remove NaN values from lists
monthly_grouped["lowest_latitude"] = monthly_grouped["lowest_latitude"].apply(lambda x: [val for val in x if not np.isnan(val)])

monthly_grouped = monthly_grouped.explode("lowest_latitude")
monthly_grouped["lowest_latitude"] = pd.to_numeric(monthly_grouped["lowest_latitude"], errors="coerce")


# Plot monthly statistics
plt.figure(figsize=(12, 6))
# Convert Period to string for plotting

sns.boxplot(x="month", 
            y="lowest_latitude", 
            data=monthly_grouped, 
            order=month_order,   
            color="lightblue",  # Set all boxes to light blue
            showmeans=True,  # Show mean line
            meanprops={"marker":"o", "markerfacecolor":"black", "markeredgecolor":"black", "markersize":8},
            boxprops=dict(edgecolor="black"),
            medianprops=dict(color="black", linewidth=2),  # Median line
            whiskerprops=dict(color="black", linestyle="-"),  # Whiskers
            capprops=dict(color="black"),  # Caps
            flierprops=dict(marker="o", color="green", alpha=0.5) 
            )
                 
plt.ylabel("Latitudes(N)", fontsize=20, weight='bold')
plt.title("RWB latitudinal extent (1980-2019)",fontsize=20, weight='bold')
plt.xticks(fontsize=15, weight='bold', rotation=90)
plt.yticks(fontsize=15, weight='bold')

# Create custom legend
legend_patches = [
    mpatches.Patch(facecolor="lightblue", edgecolor="black", label="Interquartile Range (IQR)"),
    mlines.Line2D([], [], color="black", linewidth=2, label="Median"),  # Median Line
    mlines.Line2D([], [], color="black", marker="o", linestyle="None", markersize=8, label="Mean"),  # Mean Marker
    mlines.Line2D([], [], color="black", marker="o", linestyle="None", markersize=8, alpha=0.5, label="Outliers")  # Outliers
]

# Add legend
#plt.legend(handles=legend_patches, loc="lower right", fontsize=12)
plt.legend(
    handles=legend_patches,
    loc="center left", 
    bbox_to_anchor=(1.02, 0.5),  # Pushes legend to the right outside the plot
    fontsize=12,
    borderaxespad=0
)
plt.tight_layout()
plt.savefig(r"D:\IITB_Project\Plots\Multiple_level_analysis\latitudinal_intrusions.png", dpi=300)
plt.show()


